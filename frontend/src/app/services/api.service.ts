import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface QueryResponse {
  answer: string;
  sources: string[];
  conversation_id?: string;
  user_message_id?: string;
  assistant_message_id?: string;
}

export interface ConversationMatch {
  where: 'title' | 'message';
  snippet: string;
  message_id?: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  pinned: boolean;
  archived: boolean;
  message_count?: number;
  match?: ConversationMatch;  // só presente em resultados de pesquisa
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sources: string[];
  tools_used: any[];
  tokens_input: number | null;
  tokens_output: number | null;
  agent_actions: any[];
  execution_state: string | null;
  metadata: Record<string, any>;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

export interface GroupedConversations {
  pinned: Conversation[];
  today: Conversation[];
  yesterday: Conversation[];
  last_7_days: Conversation[];
  last_30_days: Conversation[];
  older: Conversation[];
}

export interface AutoTitleResponse {
  title: string;
  conversation: Conversation;
}

export interface ColumnInfo {
  key: string;
  type: string;
  samples: string[];
}

export interface UploadResponse {
  filename: string;
  chunks_added: number;
  total_chunks: number;
  extraction_mode: string;
  row_count: number;
  columns: ColumnInfo[];
}

export interface DocumentInfo {
  filename: string;
  size: number;
  chunks: number;
  mode: string;
  modified_at: string;
}

export interface DeleteResponse {
  deleted: string;
  total_chunks: number;
}

export interface RenameResponse {
  old: string;
  renamed: string;
}

export interface ReextractResponse {
  reextracted: string;
  mode: string;
  chunks_added: number;
  total_chunks: number;
}

export interface LlmHealth {
  ok: boolean;
  provider: string;            // "ollama" | "groq"
  url?: string;
  configured_model: string;
  available_models?: string[];
  model_available?: boolean;
  error?: string;
  detail?: string;
  hint?: string;
  keep_alive?: string;
}

export interface EmbedHealth {
  ok: boolean;
  provider: string;            // "local" | "gemini"
  configured_model: string;
  dimensions?: number;
  collection?: string;
  chunks_indexed?: number;
  error?: string | null;
  hint?: string | null;
  note?: string;
}

export interface AdminConfig {
  llm_provider: string;
  ollama_model: string;
  ollama_url: string;
  groq_model: string;
  groq_api_key_set: boolean;
  groq_api_key_preview: string;
  gemini_llm_model: string;
  gemini_llm_api_key_set: boolean;
  gemini_llm_api_key_preview: string;
  embed_provider: string;
  google_api_key_set: boolean;
  google_api_key_preview: string;
}

export interface ScraperEntry {
  id: string;
  url: string;
  label: string;
  max_pages: number;
  pages_crawled: number;
  last_scraped: string | null;
  status: 'pending' | 'ok' | 'error';
  chunks_indexed: number;
  error: string | null;
}

export interface AdminConfigUpdate {
  llm_provider?: string;
  ollama_model?: string;
  ollama_url?: string;
  groq_model?: string;
  groq_api_key?: string;
  gemini_llm_model?: string;
  gemini_llm_api_key?: string;
  embed_provider?: string;
  google_api_key?: string;
}

export interface ScrapingStatus {
  enabled: boolean;
  web_chunks: number;
}

export interface ScrapingToggleResponse {
  enabled: boolean;
  web_chunks: number;
  removed_chunks?: number;
  total_chunks: number;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  query(question: string, conversationId?: string | null): Observable<QueryResponse> {
    const body: { question: string; conversation_id?: string } = { question };
    if (conversationId) {
      body.conversation_id = conversationId;
    }
    return this.http.post<QueryResponse>(`${this.baseUrl}/query`, body);
  }

  uploadPdf(file: File): Observable<UploadResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<UploadResponse>(`${this.baseUrl}/upload`, form);
  }

  listDocuments(): Observable<DocumentInfo[]> {
    return this.http.get<DocumentInfo[]>(`${this.baseUrl}/documents`);
  }

  deleteDocument(filename: string): Observable<DeleteResponse> {
    return this.http.delete<DeleteResponse>(
      `${this.baseUrl}/documents/${encodeURIComponent(filename)}`
    );
  }

  getDocumentUrl(filename: string): string {
    return `${this.baseUrl}/documents/${encodeURIComponent(filename)}/file`;
  }

  renameDocument(oldFilename: string, newFilename: string): Observable<RenameResponse> {
    return this.http.patch<RenameResponse>(
      `${this.baseUrl}/documents/${encodeURIComponent(oldFilename)}/rename`,
      { new_filename: newFilename }
    );
  }

  llmHealth(): Observable<LlmHealth> {
    return this.http.get<LlmHealth>(`${this.baseUrl}/health/llm`);
  }

  embedHealth(): Observable<EmbedHealth> {
    return this.http.get<EmbedHealth>(`${this.baseUrl}/health/embeddings`);
  }

  reextractDocument(filename: string, mode: 'hybrid' | 'markdown' | 'text' | 'table' | 'vision'): Observable<ReextractResponse> {
    return this.http.post<ReextractResponse>(
      `${this.baseUrl}/documents/${encodeURIComponent(filename)}/reextract`,
      { mode }
    );
  }

  // -------------------- Conversas --------------------

  createConversation(title?: string): Observable<Conversation> {
    return this.http.post<Conversation>(`${this.baseUrl}/conversations`, { title });
  }

  /** Lista plana ordenada por (pinned DESC, updated_at DESC). */
  listConversations(): Observable<Conversation[]> {
    return this.http.get<Conversation[]>(`${this.baseUrl}/conversations`);
  }

  /** Lista agrupada por tempo — pronta para o sidebar. */
  listConversationsGrouped(): Observable<GroupedConversations> {
    return this.http.get<GroupedConversations>(`${this.baseUrl}/conversations?grouped=true`);
  }

  getConversation(id: string, includeMessages = true): Observable<ConversationWithMessages> {
    return this.http.get<ConversationWithMessages>(
      `${this.baseUrl}/conversations/${encodeURIComponent(id)}?include_messages=${includeMessages}`
    );
  }

  updateConversation(
    id: string,
    patch: { title?: string; pinned?: boolean; archived?: boolean }
  ): Observable<Conversation> {
    return this.http.patch<Conversation>(
      `${this.baseUrl}/conversations/${encodeURIComponent(id)}`,
      patch
    );
  }

  deleteConversation(id: string): Observable<{ deleted: string }> {
    return this.http.delete<{ deleted: string }>(
      `${this.baseUrl}/conversations/${encodeURIComponent(id)}`
    );
  }

  searchConversations(q: string, limit = 50): Observable<Conversation[]> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return this.http.get<Conversation[]>(
      `${this.baseUrl}/conversations/search?${params.toString()}`
    );
  }

  autoTitleConversation(id: string): Observable<AutoTitleResponse> {
    return this.http.post<AutoTitleResponse>(
      `${this.baseUrl}/conversations/${encodeURIComponent(id)}/auto-title`,
      {}
    );
  }

  // -------------------- Admin --------------------

  getAdminConfig(): Observable<AdminConfig> {
    return this.http.get<AdminConfig>(`${this.baseUrl}/admin/config`);
  }

  updateAdminConfig(cfg: AdminConfigUpdate): Observable<{ updated: string[]; config: AdminConfig }> {
    return this.http.post<{ updated: string[]; config: AdminConfig }>(
      `${this.baseUrl}/admin/config`, cfg
    );
  }

  // -------------------- Scrapers --------------------

  listScrapers(): Observable<ScraperEntry[]> {
    return this.http.get<ScraperEntry[]>(`${this.baseUrl}/admin/scrapers`);
  }

  addScraper(url: string, label: string, maxPages: number = 100): Observable<ScraperEntry> {
    return this.http.post<ScraperEntry>(`${this.baseUrl}/admin/scrapers`, {
      url,
      label,
      max_pages: maxPages,
    });
  }

  deleteScraper(id: string): Observable<{ deleted: string; url: string }> {
    return this.http.delete<{ deleted: string; url: string }>(
      `${this.baseUrl}/admin/scrapers/${encodeURIComponent(id)}`
    );
  }

  runScraper(id: string): Observable<ScraperEntry> {
    return this.http.post<ScraperEntry>(
      `${this.baseUrl}/admin/scrapers/${encodeURIComponent(id)}/run`, {}
    );
  }

  // -------------------- Scraping toggle --------------------

  getScrapingStatus(): Observable<ScrapingStatus> {
    return this.http.get<ScrapingStatus>(`${this.baseUrl}/admin/scraping/status`);
  }

  enableScraping(): Observable<ScrapingToggleResponse> {
    return this.http.post<ScrapingToggleResponse>(`${this.baseUrl}/admin/scraping/enable`, {});
  }

  disableScraping(): Observable<ScrapingToggleResponse> {
    return this.http.post<ScrapingToggleResponse>(`${this.baseUrl}/admin/scraping/disable`, {});
  }
}
