import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Bot, User, Sparkles, MessageSquarePlus,
  Globe, Database, ExternalLink, KeyRound, Brain, Trash2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './index.css';

const DEFAULT_USER_ID = 'user_id';
const ENV_TOKEN = import.meta.env.VITE_RAG_API_TOKEN || '';
const MODEL_OPTIONS = [
  { value: 'global.anthropic.claude-haiku-4-5-20251001-v1:0', label: 'Haiku 4.5' },
  { value: 'anthropic.claude-sonnet-5', label: 'Sonnet 5' },
  { value: 'anthropic.claude-opus-4-8', label: 'Opus 4.8' },
];

function authHeaders(extra = {}) {
  const token = ENV_TOKEN || window.localStorage.getItem('rag_api_token') || '';
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [limit, setLimit] = useState(20);
  const [webSearch, setWebSearch] = useState(false);
  const [model, setModel] = useState(MODEL_OPTIONS[0].value);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [thinkingLevel, setThinkingLevel] = useState('medium');
  const [apiToken, setApiToken] = useState(() => ENV_TOKEN || window.localStorage.getItem('rag_api_token') || '');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const userId = DEFAULT_USER_ID;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`/api/conversations?user_id=${encodeURIComponent(userId)}`, {
          headers: authHeaders()
        });
        if (response.ok) {
          const data = await response.json();
          setChatHistory(data);
        }
      } catch (err) {
        console.error('Failed to fetch history:', err);
      }
    };
    fetchHistory();
  }, [apiToken, userId]);

  const saveApiToken = (value) => {
    setApiToken(value);
    if (value) {
      window.localStorage.setItem('rag_api_token', value);
    } else {
      window.localStorage.removeItem('rag_api_token');
    }
  };

  const loadConversation = async (id) => {
    if (id === conversationId) return;
    setConversationId(id);
    setMessages([]);
    setIsLoading(true);
    try {
      const response = await fetch(`/api/conversations/${id}?user_id=${encodeURIComponent(userId)}`, {
        headers: authHeaders()
      });
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
      }
    } catch (err) {
      console.error('Failed to load conversation:', err);
    } finally {
      setIsLoading(false);
    }
  };


  const deleteConversation = async (id) => {
    const shouldDelete = window.confirm('Delete this conversation?');
    if (!shouldDelete) return;

    try {
      const response = await fetch(`/api/conversations/${id}?user_id=${encodeURIComponent(userId)}`, {
        method: 'DELETE',
        headers: authHeaders()
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Delete failed');
      }
      setChatHistory(prev => prev.filter(chat => chat.id !== id));
      if (id === conversationId) {
        setConversationId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      setMessages(prev => [...prev, { role: 'assistant', content: `Could not delete the conversation: ${err.message}` }]);
    }
  };

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    const userMessage = { role: 'user', content: question };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          message: question,
          user_id: userId,
          limit: Math.min(Math.max(Number(limit) || 1, 1), 25),
          web_search: webSearch,
          web_search_limit: webSearch ? 20 : 1,
          model,
          thinking_enabled: thinkingEnabled,
          thinking_level: thinkingLevel,
          conversation_id: conversationId,
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Network response was not ok');
      }

      const data = await response.json();

      if (!conversationId && data.conversation_id) {
        setConversationId(data.conversation_id);
        setChatHistory(prev => [
          { id: data.conversation_id, title: question.substring(0, 30) + (question.length > 30 ? '...' : ''), date: 'Just now' },
          ...prev
        ]);
      }

      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          citations: data.citations || [],
          follow_up_questions: data.follow_up_questions || []
        }
      ]);
    } catch (error) {
      console.error('Error fetching chat:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: `Sorry, I encountered an error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setConversationId(null);
  };

  const handleFollowUpQuestion = (question) => {
    setInput(question);
    inputRef.current?.focus();
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-icon">
            <Sparkles size={18} />
          </div>
          <h1 className="app-title">News RAG</h1>
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <MessageSquarePlus size={18} />
          New Chat
        </button>

        {!ENV_TOKEN && (
          <div className="token-box">
            <label className="token-label" htmlFor="rag-token">
              <KeyRound size={14} /> API token
            </label>
            <input
              id="rag-token"
              className="token-input"
              type="password"
              value={apiToken}
              onChange={(e) => saveApiToken(e.target.value)}
              placeholder="Bearer token"
              autoComplete="off"
            />
          </div>
        )}

        <div className="history-list">
          <div className="history-section-title">Recent Chats</div>
          {chatHistory.map(chat => (
            <div key={chat.id} className={`history-row ${chat.id === conversationId ? 'active' : ''}`}>
              <button
                type="button"
                className="history-item"
                onClick={() => loadConversation(chat.id)}
              >
                <MessageSquarePlus size={16} />
                <span className="history-text">{chat.title}</span>
              </button>
              <button
                type="button"
                className="delete-chat-btn"
                onClick={() => deleteConversation(chat.id)}
                aria-label={`Delete ${chat.title}`}
                title="Delete conversation"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-content">
        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <Sparkles size={32} color="#6366f1" />
              </div>
              <h2>How can I help you today?</h2>
              <p>Ask a question about indexed news, or enable Web Search for live citations.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={`${msg.role}-${idx}`} className="message-wrapper">
                <div className="message">
                  <div className={`avatar ${msg.role === 'user' ? 'user' : 'ai'}`}>
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className="message-content">
                    <div className="message-author">
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </div>
                    <div className="message-text markdown-body">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>

                    {msg.follow_up_questions && msg.follow_up_questions.length > 0 && (
                      <div className="follow-ups-box">
                        <div className="follow-ups-title">Follow-up questions</div>
                        <div className="follow-ups-list">
                          {msg.follow_up_questions.map((question, i) => (
                            <button type="button" key={`${question}-${i}`} className="follow-up-chip" onClick={() => handleFollowUpQuestion(question)}>
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-box">
                        <div className="citations-title">
                          <Database size={14} /> Sources
                        </div>
                        {msg.citations.map((cite, i) => (
                          <div key={`${cite.citation_marker || i}-${cite.article_id || cite.web_search_id || i}`} className="citation-item">
                            <span className="citation-marker">{cite.citation_marker}</span>
                            <div className="citation-body">
                              {cite.url ? (
                                <a href={cite.url} target="_blank" rel="noreferrer" className="citation-link">
                                  {cite.title || 'Untitled source'} <ExternalLink size={12} opacity={0.6} />
                                </a>
                              ) : (
                                <span className="citation-link">{cite.title || 'Untitled source'}</span>
                              )}
                              <div className="citation-meta">
                                {cite.provider || 'unknown'}{cite.published_at ? ` • ${cite.published_at}` : ''}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="message-wrapper">
              <div className="message">
                <div className="avatar ai">
                  <Bot size={20} />
                </div>
                <div className="message-content">
                  <div className="message-author">Assistant</div>
                  <div className="loading-dots">
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-wrapper">
          <div className="input-container">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Message News RAG..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows="1"
            />
            <div className="input-actions">
              <div className="settings-row">
                <label className="setting-item model-setting" title="Main chat model">
                  <Bot size={14} />
                  <select
                    className="styled-select model-select"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    aria-label="Main chat model"
                  >
                    {MODEL_OPTIONS.map(option => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </label>

                <label className={`setting-item web-search-toggle ${webSearch ? 'active' : ''}`}>
                  <Globe size={16} color={webSearch ? '#6366f1' : '#a1a5b0'} />
                  <span>Web Search</span>
                  <span className="toggle-switch small-toggle">
                    <input
                      type="checkbox"
                      checked={webSearch}
                      onChange={(e) => setWebSearch(e.target.checked)}
                      aria-label="Enable web search"
                    />
                    <span className="slider"></span>
                  </span>
                </label>

                <label className={`setting-item web-search-toggle ${thinkingEnabled ? 'active' : ''}`}>
                  <Brain size={16} color={thinkingEnabled ? '#6366f1' : '#a1a5b0'} />
                  <span>Thinking</span>
                  <span className="toggle-switch small-toggle">
                    <input
                      type="checkbox"
                      checked={thinkingEnabled}
                      onChange={(e) => setThinkingEnabled(e.target.checked)}
                      aria-label="Enable thinking"
                    />
                    <span className="slider"></span>
                  </span>
                </label>

                {thinkingEnabled && (
                  <label className="setting-item" title="Thinking token budget">
                    <select
                      className="styled-select compact-select"
                      value={thinkingLevel}
                      onChange={(e) => setThinkingLevel(e.target.value)}
                      aria-label="Thinking level"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="max">Max</option>
                    </select>
                  </label>
                )}

                <label className="setting-item" title="Max articles to retrieve">
                  <Database size={14} />
                  <input
                    type="number"
                    className="styled-input compact-input"
                    value={limit}
                    onChange={(e) => setLimit(e.target.value)}
                    min="1"
                    max="25"
                    aria-label="Max articles to retrieve"
                  />
                </label>
              </div>
              <button
                className="send-btn"
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                aria-label="Send message"
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
