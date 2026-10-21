import { Component, HostListener, OnDestroy, OnInit } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import {
  AdminConfig,
  AdminConfigUpdate,
  ApiService,
  DocumentInfo,
  ScraperEntry,
  ScrapingStatus,
  UploadResponse,
} from '../services/api.service';

interface BatchFileEntry {
  file: File;
  status: 'pending' | 'uploading' | 'done' | 'error';
  result?: UploadResponse;
  error?: string;
}

@Component({
  selector: 'app-admin',
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css'],
})
export class AdminComponent implements OnInit, OnDestroy {
  // ---- Admin config ----
  adminConfig: AdminConfig | null = null;
  adminConfigLoading = false;
  adminConfigSaving = false;
  adminConfigSaved = false;
  adminConfigError = '';

  configLlmProvider = 'ollama';
  configOllamaModel = '';
  configOllamaUrl = '';
  configGroqModel = '';
  configGroqApiKey = '';
  configGeminiLlmModel = '';
  configGeminiLlmApiKey = '';
  configEmbedProvider = 'local';
  configGoogleApiKey = '';

  isDragOver = false;
  activeTab: 'upload' | 'documents' | 'config' | 'scraping' = 'upload';

  darkMode = false;
  private static readonly DARK_MODE_KEY = 'educaassistai.darkMode';

  // ---- Batch upload ----
  batchFiles: BatchFileEntry[] = [];
  batchUploading = false;

  // ---- Documents ----
  documents: DocumentInfo[] = [];
  documentsLoading = false;

  get indexedDocuments(): DocumentInfo[] {
    return this.documents.filter(d => d.chunks > 0);
  }

  get totalIndexedChunks(): number {
    return this.indexedDocuments.reduce((sum, d) => sum + d.chunks, 0);
  }
  documentsError = '';
  deletingFilename: string | null = null;
  editingFilename: string | null = null;
  editingNameDraft = '';
  renamingFilename: string | null = null;
  reextractingFilename: string | null = null;

  actionStatus = '';
  actionError = false;

  // ---- Preview ----
  previewFilename: string | null = null;
  previewUrl: SafeResourceUrl | null = null;

  // ---- Scrapers ----
  scrapers: ScraperEntry[] = [];
  scrapersLoading = false;
  scrapersError = '';
  scraperNewUrl = '';
  scraperNewLabel = '';
  scraperNewMaxPages: number = 100;
  scraperAdding = false;
  scraperAddError = '';
  runningScraperId: string | null = null;
  deletingScraperId: string | null = null;

  // ---- Scraping toggle ----
  scrapingEnabled: boolean | null = null;
  scrapingWebChunks = 0;
  scrapingToggling = false;
  scrapingToggleError = '';

  private destroy$ = new Subject<void>();

  constructor(
    private api: ApiService,
    private sanitizer: DomSanitizer,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.loadDarkModePreference();
    this.loadAdminConfig();
    this.refreshDocuments();
    this.refreshScrapers();
    this.loadScrapingStatus();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  @HostListener('window:keydown', ['$event'])
  onGlobalKey(event: KeyboardEvent): void {
    if (event.key === 'Escape' && this.previewUrl) {
      this.closePreview();
    }
  }

  goBack(): void {
    this.router.navigate(['/']);
  }

  toggleDarkMode(): void {
    this.darkMode = !this.darkMode;
    try { localStorage.setItem(AdminComponent.DARK_MODE_KEY, this.darkMode ? '1' : '0'); } catch {}
    this.applyDarkMode();
  }

  private loadDarkModePreference(): void {
    let preferred = false;
    try {
      const saved = localStorage.getItem(AdminComponent.DARK_MODE_KEY);
      if (saved !== null) {
        preferred = saved === '1';
      } else if (window.matchMedia) {
        preferred = window.matchMedia('(prefers-color-scheme: dark)').matches;
      }
    } catch { }
    this.darkMode = preferred;
    this.applyDarkMode();
  }

  private applyDarkMode(): void {
    if (typeof document === 'undefined') { return; }
    if (this.darkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }

  // ---- Admin config ----

  loadAdminConfig(): void {
    this.adminConfigLoading = true;
    this.api.getAdminConfig().pipe(takeUntil(this.destroy$)).subscribe({
      next: cfg => {
        this.adminConfig = cfg;
        this.configLlmProvider = cfg.llm_provider;
        this.configOllamaModel = cfg.ollama_model;
        this.configOllamaUrl = cfg.ollama_url;
        this.configGroqModel = cfg.groq_model;
        this.configGeminiLlmModel = cfg.gemini_llm_model;
        this.configEmbedProvider = cfg.embed_provider;
        this.adminConfigLoading = false;
      },
      error: () => {
        this.adminConfigLoading = false;
        this.adminConfigError = 'Não foi possível carregar a configuração do backend.';
      }
    });
  }

  saveAdminConfig(): void {
    this.adminConfigSaving = true;
    this.adminConfigSaved = false;
    this.adminConfigError = '';

    const payload: AdminConfigUpdate = {
      llm_provider: this.configLlmProvider,
      ollama_model: this.configOllamaModel,
      ollama_url: this.configOllamaUrl,
      groq_model: this.configGroqModel,
      gemini_llm_model: this.configGeminiLlmModel,
      embed_provider: this.configEmbedProvider,
    };
    if (this.configGroqApiKey.trim()) {
      payload.groq_api_key = this.configGroqApiKey.trim();
    }
    if (this.configGeminiLlmApiKey.trim()) {
      payload.gemini_llm_api_key = this.configGeminiLlmApiKey.trim();
    }
    if (this.configGoogleApiKey.trim()) {
      payload.google_api_key = this.configGoogleApiKey.trim();
    }

    this.api.updateAdminConfig(payload).pipe(takeUntil(this.destroy$)).subscribe({
      next: result => {
        this.adminConfig = result.config;
        this.adminConfigSaving = false;
        this.adminConfigSaved = true;
        this.configGroqApiKey = '';
        this.configGeminiLlmApiKey = '';
        this.configGoogleApiKey = '';
        setTimeout(() => { this.adminConfigSaved = false; }, 3000);
      },
      error: () => {
        this.adminConfigSaving = false;
        this.adminConfigError = 'Erro ao guardar configuração.';
      }
    });
  }

  // ---- Batch upload ----

  onBatchFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files) { return; }
    const ACCEPTED = ['.pdf', '.docx', '.html'];
    Array.from(input.files)
      .filter(f => ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext)))
      .forEach(f => {
        if (!this.batchFiles.find(e => e.file.name === f.name && e.file.size === f.size)) {
          this.batchFiles.push({ file: f, status: 'pending' });
        }
      });
    input.value = '';
  }

  onBatchDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragOver = false;
    const files = event.dataTransfer?.files;
    if (!files) { return; }
    const ACCEPTED = ['.pdf', '.docx', '.html'];
    Array.from(files)
      .filter(f => ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext)))
      .forEach(f => {
        if (!this.batchFiles.find(e => e.file.name === f.name && e.file.size === f.size)) {
          this.batchFiles.push({ file: f, status: 'pending' });
        }
      });
  }

  removeBatchFile(index: number): void {
    this.batchFiles.splice(index, 1);
  }

  hasPendingFiles(): boolean {
    return this.batchFiles.some(e => e.status === 'pending' || e.status === 'error');
  }

  pendingCount(): number {
    return this.batchFiles.filter(e => e.status === 'pending' || e.status === 'error').length;
  }

  startBatchUpload(): void {
    if (this.batchUploading) { return; }
    const pending = this.batchFiles.filter(e => e.status === 'pending' || e.status === 'error');
    if (!pending.length) { return; }
    this.batchUploading = true;
    this._processNextBatchFile(pending, 0);
  }

  private _processNextBatchFile(files: BatchFileEntry[], index: number): void {
    if (index >= files.length) {
      this.batchUploading = false;
      this.refreshDocuments();
      return;
    }
    const entry = files[index];
    entry.status = 'uploading';
    this.api.uploadPdf(entry.file).pipe(takeUntil(this.destroy$)).subscribe({
      next: result => {
        entry.status = 'done';
        entry.result = result;
        this._processNextBatchFile(files, index + 1);
      },
      error: (err: any) => {
        entry.status = 'error';
        entry.error = err?.error?.detail || 'Erro ao carregar ficheiro.';
        this._processNextBatchFile(files, index + 1);
      }
    });
  }

  batchStatusLabel(status: string): string {
    const labels: Record<string, string> = {
      pending: 'Pendente',
      uploading: 'A carregar...',
      done: 'Concluído',
      error: 'Erro',
    };
    return labels[status] ?? status;
  }

  // ---- Documents ----

  refreshDocuments(): void {
    this.documentsLoading = true;
    this.documentsError = '';
    this.api.listDocuments().pipe(takeUntil(this.destroy$)).subscribe({
      next: docs => {
        this.documents = docs;
        this.documentsLoading = false;
      },
      error: () => {
        this.documentsError = 'Não foi possível obter a lista de documentos.';
        this.documentsLoading = false;
      }
    });
  }

  deleteDocument(doc: DocumentInfo): void {
    const ok = window.confirm(
      `Tem a certeza que pretende apagar "${doc.filename}"?\n\n` +
      `Esta ação remove o ficheiro e os respetivos excertos do índice. Não é reversível.`
    );
    if (!ok) { return; }
    this.deletingFilename = doc.filename;
    this.api.deleteDocument(doc.filename).pipe(takeUntil(this.destroy$)).subscribe({
      next: response => {
        this.deletingFilename = null;
        this.showActionStatus(`${response.deleted} apagado (total: ${response.total_chunks} excertos).`, false);
        if (this.previewFilename === doc.filename) { this.closePreview(); }
        this.refreshDocuments();
      },
      error: err => {
        this.deletingFilename = null;
        const detail = err?.error?.detail || (err?.status === 0 ? 'sem resposta do backend' : `HTTP ${err?.status || '?'}`);
        this.showActionStatus(`Falha ao apagar ${doc.filename}: ${detail}`, true);
      }
    });
  }

  startRename(doc: DocumentInfo): void {
    this.editingFilename = doc.filename;
    this.editingNameDraft = doc.filename.replace(/\.pdf$/i, '');
  }

  cancelRename(): void {
    this.editingFilename = null;
    this.editingNameDraft = '';
  }

  saveRename(doc: DocumentInfo): void {
    const draft = (this.editingNameDraft || '').trim();
    if (!draft) {
      this.showActionStatus('O nome não pode ficar em branco.', true);
      return;
    }
    const newName = draft.toLowerCase().endsWith('.pdf') ? draft : `${draft}.pdf`;
    if (newName === doc.filename) { this.cancelRename(); return; }
    this.renamingFilename = doc.filename;
    this.api.renameDocument(doc.filename, newName).pipe(takeUntil(this.destroy$)).subscribe({
      next: response => {
        this.renamingFilename = null;
        this.showActionStatus(`${response.old} → ${response.renamed}`, false);
        if (this.previewFilename === response.old) { this.previewFilename = response.renamed; }
        this.cancelRename();
        this.refreshDocuments();
      },
      error: err => {
        this.renamingFilename = null;
        const detail = err?.error?.detail || (err?.status === 0 ? 'sem resposta do backend' : `HTTP ${err?.status || '?'}`);
        this.showActionStatus(`Falha ao renomear: ${detail}`, true);
      }
    });
  }

  reextractDocument(doc: DocumentInfo, mode: 'hybrid' | 'markdown' | 'text' | 'table' | 'vision'): void {
    const labels: Record<string, string> = {
      hybrid: 'híbrido (texto + tabelas)',
      markdown: 'markdown estruturado',
      text: 'texto corrido',
      table: 'tabela',
      vision: 'Gemini Vision (multimodal)',
    };
    const visionNote = mode === 'vision'
      ? '\n\nUsa Gemini 1.5 Flash para processar cada página como imagem.\nPode demorar alguns minutos em PDFs longos.'
      : '';
    const ok = window.confirm(
      `Re-extrair "${doc.filename}" como ${labels[mode] || mode}?\n\n` +
      `Os ${doc.chunks} excertos atuais vão ser substituídos.` +
      visionNote
    );
    if (!ok) { return; }
    this.reextractingFilename = doc.filename;
    this.api.reextractDocument(doc.filename, mode).pipe(takeUntil(this.destroy$)).subscribe({
      next: response => {
        this.reextractingFilename = null;
        this.showActionStatus(`${response.reextracted} re-extraído como ${response.mode} (${response.chunks_added} excertos).`, false);
        this.refreshDocuments();
      },
      error: err => {
        this.reextractingFilename = null;
        const detail = err?.error?.detail || (err?.status === 0 ? 'sem resposta do backend' : `HTTP ${err?.status || '?'}`);
        this.showActionStatus(`Falha ao re-extrair ${doc.filename}: ${detail}`, true);
      }
    });
  }

  // ---- Scraping toggle ----

  loadScrapingStatus(): void {
    this.api.getScrapingStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (s: ScrapingStatus) => {
        this.scrapingEnabled = s.enabled;
        this.scrapingWebChunks = s.web_chunks;
      },
      error: () => { this.scrapingEnabled = true; }
    });
  }

  toggleScraping(enable: boolean): void {
    this.scrapingToggling = true;
    this.scrapingToggleError = '';
    const call = enable ? this.api.enableScraping() : this.api.disableScraping();
    call.pipe(takeUntil(this.destroy$)).subscribe({
      next: res => {
        this.scrapingEnabled = res.enabled;
        this.scrapingWebChunks = res.web_chunks;
        this.scrapingToggling = false;
        if (!enable && res.removed_chunks !== undefined) {
          this.showActionStatus(
            `Web scraping desativado — ${res.removed_chunks} chunk${res.removed_chunks === 1 ? '' : 's'} web removidos.`,
            false
          );
        } else if (enable) {
          this.showActionStatus('Web scraping ativado. Execute os scrapers para re-indexar.', false);
        }
      },
      error: err => {
        this.scrapingToggling = false;
        this.scrapingToggleError = err?.error?.detail || 'Erro ao alterar estado do scraping.';
      }
    });
  }

  // ---- Scrapers ----

  refreshScrapers(): void {
    this.scrapersLoading = true;
    this.scrapersError = '';
    this.api.listScrapers().pipe(takeUntil(this.destroy$)).subscribe({
      next: list => {
        this.scrapers = list;
        this.scrapersLoading = false;
      },
      error: () => {
        this.scrapersError = 'Não foi possível carregar a lista de scrapers.';
        this.scrapersLoading = false;
      }
    });
  }

  addScraper(): void {
    const url = this.scraperNewUrl.trim();
    if (!url) { return; }
    this.scraperAdding = true;
    this.scraperAddError = '';
    this.api.addScraper(url, this.scraperNewLabel.trim(), this.scraperNewMaxPages).pipe(takeUntil(this.destroy$)).subscribe({
      next: entry => {
        if (!this.scrapers.find(s => s.id === entry.id)) {
          this.scrapers.push(entry);
        }
        this.scraperNewUrl = '';
        this.scraperNewLabel = '';
        this.scraperNewMaxPages = 100;
        this.scraperAdding = false;
      },
      error: err => {
        this.scraperAdding = false;
        this.scraperAddError = err?.error?.detail || 'Erro ao adicionar URL.';
      }
    });
  }

  runScraper(entry: ScraperEntry): void {
    this.runningScraperId = entry.id;
    this.api.runScraper(entry.id).pipe(takeUntil(this.destroy$)).subscribe({
      next: updated => {
        const idx = this.scrapers.findIndex(e => e.id === entry.id);
        if (idx >= 0) { this.scrapers[idx] = updated; }
        this.runningScraperId = null;
      },
      error: err => {
        this.runningScraperId = null;
        const detail = err?.error?.detail || 'Erro ao executar scraper.';
        this.showActionStatus(`Erro scraper: ${detail}`, true);
      }
    });
  }

  deleteScraper(entry: ScraperEntry): void {
    const ok = window.confirm(
      `Remover "${entry.label || entry.url}"?\n\nOs excertos indexados desta URL serão apagados.`
    );
    if (!ok) { return; }
    this.deletingScraperId = entry.id;
    this.api.deleteScraper(entry.id).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => {
        this.scrapers = this.scrapers.filter(e => e.id !== entry.id);
        this.deletingScraperId = null;
      },
      error: err => {
        this.deletingScraperId = null;
        const detail = err?.error?.detail || 'Erro ao remover scraper.';
        this.showActionStatus(`Erro scraper: ${detail}`, true);
      }
    });
  }

  scraperStatusLabel(s: ScraperEntry): string {
    if (s.status === 'ok') {
      const pages = s.pages_crawled || 0;
      return `OK · ${pages} pág. · ${s.chunks_indexed} excertos`;
    }
    if (s.status === 'error') {
      const err = s.error || 'desconhecido';
      return `Erro: ${err.length > 80 ? err.slice(0, 80) + '…' : err}`;
    }
    return 'Pendente';
  }

  scraperErrorFull(s: ScraperEntry): string {
    return s.error || '';
  }

  // ---- Preview ----

  viewDocument(doc: DocumentInfo): void {
    const url = this.api.getDocumentUrl(doc.filename);
    this.previewFilename = doc.filename;
    this.previewUrl = this.sanitizer.bypassSecurityTrustResourceUrl(url);
  }

  openDocumentInTab(doc: DocumentInfo): void {
    window.open(this.api.getDocumentUrl(doc.filename), '_blank', 'noopener');
  }

  closePreview(): void {
    this.previewFilename = null;
    this.previewUrl = null;
  }

  // ---- Utilities ----

  private showActionStatus(msg: string, isError: boolean): void {
    this.actionStatus = msg;
    this.actionError = isError;
    setTimeout(() => { if (this.actionStatus === msg) { this.actionStatus = ''; } }, 5000);
  }

  formatSize(bytes: number): string {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  formatDate(iso: string): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '—';
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
}
