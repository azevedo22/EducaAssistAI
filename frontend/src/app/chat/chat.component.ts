import { AfterViewChecked, Component, ElementRef, HostListener, OnDestroy, OnInit, ViewChild, ViewEncapsulation } from '@angular/core';
import { Router } from '@angular/router';
import { merge, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import {
  ApiService,
  Conversation,
  EmbedHealth,
  GroupedConversations,
  LlmHealth,
  Message,
} from '../services/api.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
  encapsulation: ViewEncapsulation.None,
})
export class ChatComponent implements OnInit, AfterViewChecked, OnDestroy {
  question = '';
  answer = '';
  sources: string[] = [];
  loading = false;
  error = '';

  currentMessages: Message[] = [];
  messagesLoading = false;

  private shouldScrollToBottom = false;

  @ViewChild('messagesArea') messagesAreaRef?: ElementRef<HTMLDivElement>;
  @ViewChild('composerTextarea') composerTextareaRef?: ElementRef<HTMLTextAreaElement>;
  @ViewChild('searchInput') searchInputRef?: ElementRef<HTMLInputElement>;

  copiedMessageId: string | null = null;

  quickPrompts: string[] = [
    'Quando começam as aulas do 2.º semestre?',
    'Quem é o orientador do projeto EducaAssistAI?',
    'Como funciona a melhoria de classificação?',
    'Qual o horário de atendimento dos professores do DCES?',
  ];

  convsDrawerOpen = false;

  llmHealth: LlmHealth | null = null;
  embedHealth: EmbedHealth | null = null;

  darkMode = false;
  private static readonly DARK_MODE_KEY = 'educaassistai.darkMode';

  conversationsGrouped: GroupedConversations | null = null;
  conversationsLoading = false;
  selectedConversationId: string | null = null;

  searchQuery = '';
  searchResults: Conversation[] | null = null;
  private searchDebounceHandle: number | null = null;

  busyConversationId: string | null = null;

  private destroy$ = new Subject<void>();
  private stopQuery$ = new Subject<void>();
  private currentOptimisticMsgId: string | null = null;
  private currentQueryText = '';

  constructor(
    private api: ApiService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.refreshLlmHealth();
    this.refreshEmbedHealth();
    this.loadDarkModePreference();
    this.refreshConversations();
  }

  ngOnDestroy(): void {
    if (this.searchDebounceHandle !== null) {
      window.clearTimeout(this.searchDebounceHandle);
    }
    this.destroy$.next();
    this.destroy$.complete();
    this.stopQuery$.complete();
  }

  ngAfterViewChecked(): void {
    if (this.shouldScrollToBottom) {
      this.shouldScrollToBottom = false;
      const el = this.messagesAreaRef?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }

  @HostListener('window:keydown', ['$event'])
  onGlobalKey(event: KeyboardEvent): void {
    const mod = event.ctrlKey || event.metaKey;

    if (mod && !event.shiftKey && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.focusSearch();
      return;
    }

    if (mod && event.shiftKey && event.key.toLowerCase() === 'o') {
      event.preventDefault();
      this.newConversation();
      setTimeout(() => this.focusComposer(), 50);
      return;
    }

    if (event.key === 'Escape') {
      if (this.loading) {
        this.stopQuery();
        return;
      }
      if (this.convsDrawerOpen) {
        this.convsDrawerOpen = false;
      }
    }
  }

  goToAdmin(): void {
    this.router.navigate(['/admin']);
  }

  usePrompt(text: string): void {
    this.question = text;
    this.sendQuery();
  }

  copyMessage(m: Message, event: Event): void {
    event.stopPropagation();
    if (!navigator?.clipboard) { return; }
    navigator.clipboard.writeText(m.content).then(() => {
      this.copiedMessageId = m.id;
      setTimeout(() => {
        if (this.copiedMessageId === m.id) { this.copiedMessageId = null; }
      }, 1600);
    }).catch(() => {});
  }

  focusComposer(): void {
    const el = this.composerTextareaRef?.nativeElement;
    if (el && typeof el.focus === 'function') { el.focus(); }
  }

  focusSearch(): void {
    const el = this.searchInputRef?.nativeElement;
    if (el && typeof el.focus === 'function') {
      this.convsDrawerOpen = true;
      setTimeout(() => el.focus(), 0);
    }
  }

  toggleConvsDrawer(): void {
    this.convsDrawerOpen = !this.convsDrawerOpen;
  }

  private loadDarkModePreference(): void {
    let preferred = false;
    try {
      const saved = localStorage.getItem(ChatComponent.DARK_MODE_KEY);
      if (saved !== null) {
        preferred = saved === '1';
      } else if (window.matchMedia) {
        preferred = window.matchMedia('(prefers-color-scheme: dark)').matches;
      }
    } catch { }
    this.darkMode = preferred;
    this.applyDarkMode();
  }

  toggleDarkMode(): void {
    this.darkMode = !this.darkMode;
    try { localStorage.setItem(ChatComponent.DARK_MODE_KEY, this.darkMode ? '1' : '0'); } catch {}
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

  refreshLlmHealth(): void {
    this.api.llmHealth().pipe(takeUntil(this.destroy$)).subscribe({
      next: h => { this.llmHealth = h; },
      error: () => { this.llmHealth = null; }
    });
  }

  refreshEmbedHealth(): void {
    this.api.embedHealth().pipe(takeUntil(this.destroy$)).subscribe({
      next: h => { this.embedHealth = h; },
      error: () => { this.embedHealth = null; }
    });
  }

  refreshConversations(): void {
    this.conversationsLoading = true;
    this.api.listConversationsGrouped().pipe(takeUntil(this.destroy$)).subscribe({
      next: g => {
        this.conversationsGrouped = g;
        this.conversationsLoading = false;
      },
      error: () => {
        this.conversationsGrouped = null;
        this.conversationsLoading = false;
      }
    });
  }

  newConversation(): void {
    this.api.createConversation().pipe(takeUntil(this.destroy$)).subscribe({
      next: conv => {
        this.selectedConversationId = conv.id;
        this.question = '';
        this.answer = '';
        this.sources = [];
        this.error = '';
        this.refreshConversations();
      },
      error: () => { this.error = 'Não foi possível criar a conversa.'; }
    });
  }

  selectConversation(conv: Conversation): void {
    this.selectedConversationId = conv.id;
    this.question = '';
    this.answer = '';
    this.sources = [];
    this.error = '';
    this.loadMessagesForCurrentConversation();
    this.convsDrawerOpen = false;
    setTimeout(() => this.focusComposer(), 100);
  }

  private loadMessagesForCurrentConversation(): void {
    const id = this.selectedConversationId;
    if (!id) { this.currentMessages = []; return; }
    this.messagesLoading = true;
    this.api.getConversation(id, true).pipe(takeUntil(this.destroy$)).subscribe({
      next: conv => {
        this.currentMessages = conv.messages || [];
        this.messagesLoading = false;
        this.shouldScrollToBottom = true;
      },
      error: () => {
        this.currentMessages = [];
        this.messagesLoading = false;
      }
    });
  }

  getCurrentConversation(): Conversation | undefined {
    return this.selectedConversationId
      ? this.findConversationById(this.selectedConversationId)
      : undefined;
  }

  togglePin(conv: Conversation, event: Event): void {
    event.stopPropagation();
    this.busyConversationId = conv.id;
    this.api.updateConversation(conv.id, { pinned: !conv.pinned }).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => { this.busyConversationId = null; this.refreshConversations(); },
      error: () => { this.busyConversationId = null; }
    });
  }

  renameConversation(conv: Conversation, event: Event): void {
    event.stopPropagation();
    const novo = window.prompt('Novo título para esta conversa:', conv.title);
    if (novo === null) { return; }
    const trimmed = novo.trim();
    if (!trimmed || trimmed === conv.title) { return; }
    this.busyConversationId = conv.id;
    this.api.updateConversation(conv.id, { title: trimmed }).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => { this.busyConversationId = null; this.refreshConversations(); },
      error: () => { this.busyConversationId = null; }
    });
  }

  autoTitleConversation(conv: Conversation, event: Event): void {
    event.stopPropagation();
    this.busyConversationId = conv.id;
    this.api.autoTitleConversation(conv.id).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => { this.busyConversationId = null; this.refreshConversations(); },
      error: err => {
        this.busyConversationId = null;
        const detail = (err && err.error && err.error.detail) || 'falhou';
        this.error = `Auto-rename: ${detail}`;
      }
    });
  }

  deleteConversation(conv: Conversation, event: Event): void {
    event.stopPropagation();
    const ok = window.confirm(
      `Apagar a conversa "${conv.title}"?\n\n` +
      `As ${conv.message_count ?? 0} mensagens vão ser removidas. Não é reversível.`
    );
    if (!ok) { return; }
    this.busyConversationId = conv.id;
    this.api.deleteConversation(conv.id).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => {
        this.busyConversationId = null;
        if (this.selectedConversationId === conv.id) {
          this.selectedConversationId = null;
          this.answer = '';
          this.sources = [];
        }
        this.refreshConversations();
      },
      error: () => { this.busyConversationId = null; }
    });
  }

  onSearchInput(value: string): void {
    this.searchQuery = value;
    if (this.searchDebounceHandle !== null) { window.clearTimeout(this.searchDebounceHandle); }
    const q = value.trim();
    if (!q) { this.searchResults = null; return; }
    this.searchDebounceHandle = window.setTimeout(() => {
      this.api.searchConversations(q).pipe(takeUntil(this.destroy$)).subscribe({
        next: r => { this.searchResults = r; },
        error: () => { this.searchResults = []; }
      });
    }, 200);
  }

  clearSearch(): void {
    this.searchQuery = '';
    this.searchResults = null;
  }

  sendQuery(): void {
    const text = (this.question || '').trim();
    if (!text || this.loading) { return; }

    this.loading = true;
    this.error = '';
    this.answer = '';
    this.sources = [];
    this.currentQueryText = text;

    const optimisticUserMsg: Message = {
      id: 'optimistic-' + Date.now(),
      conversation_id: this.selectedConversationId || '',
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
      sources: [],
      tools_used: [],
      tokens_input: null,
      tokens_output: null,
      agent_actions: [],
      execution_state: 'pending',
      metadata: {},
    };
    this.currentOptimisticMsgId = optimisticUserMsg.id;
    this.currentMessages = [...this.currentMessages, optimisticUserMsg];
    this.shouldScrollToBottom = true;
    this.question = '';

    const cancel$ = merge(this.destroy$, this.stopQuery$);

    const doQuery = (conversationId: string | null) => {
      this.api.query(text, conversationId).pipe(takeUntil(cancel$)).subscribe({
        next: response => {
          this.answer = response.answer;
          this.sources = response.sources;
          this.loading = false;
          this.currentOptimisticMsgId = null;
          this.currentQueryText = '';
          if (response.conversation_id) {
            this.selectedConversationId = response.conversation_id;
            this.maybeAutoTitle(response.conversation_id);
            this.refreshConversations();
            this.loadMessagesForCurrentConversation();
          }
          setTimeout(() => this.focusComposer(), 0);
        },
        error: () => {
          this.error = 'Erro ao contactar o servidor. Verifique se o backend está a correr em http://localhost:8000.';
          this.loading = false;
          this.currentMessages = this.currentMessages.filter(m => m.id !== optimisticUserMsg.id);
          this.currentOptimisticMsgId = null;
          this.currentQueryText = '';
        }
      });
    };

    if (this.selectedConversationId) {
      doQuery(this.selectedConversationId);
    } else {
      this.api.createConversation().pipe(takeUntil(cancel$)).subscribe({
        next: conv => { this.selectedConversationId = conv.id; doQuery(conv.id); },
        error: () => { doQuery(null); }
      });
    }
  }

  stopQuery(): void {
    if (!this.loading) { return; }
    this.stopQuery$.next();
    this.loading = false;
    // Restore the question text so the user can edit and re-send
    this.question = this.currentQueryText;
    this.currentQueryText = '';
    // Remove the unconfirmed optimistic user message
    if (this.currentOptimisticMsgId) {
      this.currentMessages = this.currentMessages.filter(m => m.id !== this.currentOptimisticMsgId);
      this.currentOptimisticMsgId = null;
    }
    setTimeout(() => this.focusComposer(), 0);
  }

  private maybeAutoTitle(conversationId: string): void {
    const conv = this.findConversationById(conversationId);
    if (conv && conv.title !== 'Nova conversa') return;
    this.api.autoTitleConversation(conversationId).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => this.refreshConversations(),
      error: () => {}
    });
  }

  private findConversationById(id: string): Conversation | undefined {
    if (this.searchResults) return this.searchResults.find(c => c.id === id);
    const g = this.conversationsGrouped;
    if (!g) return undefined;
    const all: Conversation[] = [
      ...g.pinned, ...g.today, ...g.yesterday,
      ...g.last_7_days, ...g.last_30_days, ...g.older
    ];
    return all.find(c => c.id === id);
  }

  formatMessageTime(iso: string): string {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    const pad = (n: number) => n.toString().padStart(2, '0');
    const today = new Date();
    const isToday = d.toDateString() === today.toDateString();
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    if (isToday) return time;
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)} ${time}`;
  }

  formatRelativeTime(iso: string): string {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    const diffSec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (diffSec < 60) return 'agora';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} h`;
    if (diffSec < 7 * 86400) return `${Math.floor(diffSec / 86400)} d`;
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}`;
  }

  llmBadgeLabel(): string {
    if (!this.llmHealth) return '';
    const labels: Record<string, string> = { groq: 'Groq', gemini: 'Gemini', ollama: 'Ollama' };
    const provider = labels[this.llmHealth.provider] ?? this.llmHealth.provider;
    return `${provider} · ${this.llmHealth.configured_model}`;
  }

  llmBadgeTitle(): string {
    if (!this.llmHealth) return 'Estado do LLM desconhecido';
    const lines = [
      'LLM',
      `Fornecedor: ${this.llmHealth.provider}`,
      `Modelo: ${this.llmHealth.configured_model}`,
      `Estado: ${this.llmHealth.ok ? 'acessível' : 'indisponível'}`,
    ];
    if (this.llmHealth.url) lines.push(`URL: ${this.llmHealth.url}`);
    if (!this.llmHealth.ok && this.llmHealth.hint) lines.push('', this.llmHealth.hint);
    return lines.join('\n');
  }

  embedBadgeLabel(): string {
    if (!this.embedHealth) return '';
    const provider = this.embedHealth.provider === 'gemini' ? 'Gemini' : 'Local';
    const model = (this.embedHealth.configured_model || '')
      .replace(/^models\//, '')
      .replace(/^paraphrase-multilingual-/, '');
    return `Embed · ${provider} · ${model}`;
  }

  embedBadgeTitle(): string {
    if (!this.embedHealth) return 'Estado dos embeddings desconhecido';
    const lines = [
      'Embeddings',
      `Fornecedor: ${this.embedHealth.provider}`,
      `Modelo: ${this.embedHealth.configured_model}`,
      `Dimensões: ${this.embedHealth.dimensions ?? '?'}`,
      `Estado: ${this.embedHealth.ok ? 'pronto' : 'não configurado'}`,
    ];
    if (this.embedHealth.collection) {
      lines.push(`Coleção: ${this.embedHealth.collection} (${this.embedHealth.chunks_indexed ?? 0} excertos)`);
    }
    if (!this.embedHealth.ok && this.embedHealth.hint) lines.push('', this.embedHealth.hint);
    return lines.join('\n');
  }
}
