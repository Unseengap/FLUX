/**
 * FLUX Chat Interface — JavaScript
 * 
 * Handles:
 * - Model loading
 * - SSE streaming for generation
 * - Thinking process visualization
 * - Character-by-character animation
 */

class FluxChat {
    constructor() {
        this.modelLoaded = false;
        this.isGenerating = false;
        this.eventSource = null;
        
        this.init();
    }
    
    init() {
        // DOM elements
        this.loadModelBtn = document.getElementById('load-model-btn');
        this.loadModelText = document.getElementById('load-model-text');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.chatContainer = document.getElementById('chat-container');
        this.phasesContainer = document.getElementById('phases-container');
        this.thinkingStatus = document.getElementById('thinking-status');
        this.memoryPanel = document.getElementById('memory-panel');
        this.memoryResults = document.getElementById('memory-results');
        this.temperatureInput = document.getElementById('temperature');
        this.tempValue = document.getElementById('temp-value');
        this.maxLengthInput = document.getElementById('max-length');
        this.learnModeInput = document.getElementById('learn-mode');
        
        // Event listeners
        this.loadModelBtn.addEventListener('click', () => this.loadModel());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => this.updateSendButton());
        this.temperatureInput.addEventListener('input', (e) => {
            this.tempValue.textContent = e.target.value;
        });
        
        // Check if model is already loaded
        this.checkModelStatus();
    }
    
    async checkModelStatus() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.loaded) {
                this.modelLoaded = true;
                this.loadModelText.textContent = 'Model Ready';
                this.loadModelBtn.classList.add('btn-success');
                this.updateSendButton();
            }
        } catch (e) {
            console.error('Failed to check model status:', e);
        }
    }
    
    async loadModel() {
        if (this.modelLoaded) return;
        
        this.loadModelText.textContent = 'Loading...';
        this.loadModelBtn.disabled = true;
        
        try {
            const res = await fetch('/model/load', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'loaded' || data.status === 'already_loaded') {
                this.modelLoaded = true;
                this.loadModelText.textContent = 'Model Ready';
                this.loadModelBtn.style.background = 'var(--accent-success)';
                this.updateSendButton();
                
                // Show stats
                console.log('Model loaded:', data.stats);
            } else {
                throw new Error(data.message || 'Failed to load model');
            }
        } catch (e) {
            console.error('Failed to load model:', e);
            this.loadModelText.textContent = 'Load Failed';
            this.loadModelBtn.style.background = 'var(--accent-error)';
            setTimeout(() => {
                this.loadModelText.textContent = 'Retry Load';
                this.loadModelBtn.style.background = '';
                this.loadModelBtn.disabled = false;
            }, 3000);
        }
    }
    
    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText || !this.modelLoaded || this.isGenerating;
    }
    
    async sendMessage() {
        const prompt = this.messageInput.value.trim();
        if (!prompt || !this.modelLoaded || this.isGenerating) return;
        
        this.isGenerating = true;
        this.updateSendButton();
        
        // Clear welcome message if present
        const welcome = this.chatContainer.querySelector('.welcome-message');
        if (welcome) welcome.remove();
        
        // Add user message
        this.addMessage(prompt, 'user');
        this.messageInput.value = '';
        
        // Clear thinking panel
        this.clearThinkingPanel();
        this.thinkingStatus.textContent = 'Processing';
        this.thinkingStatus.classList.add('active');
        
        // Create assistant message placeholder
        const assistantMsg = this.addMessage('', 'assistant', true);
        const textSpan = assistantMsg.querySelector('.message-text');
        const metaSpan = assistantMsg.querySelector('.message-meta');
        
        // Start SSE connection
        this.startGeneration(prompt, textSpan, metaSpan);
    }
    
    startGeneration(prompt, textSpan, metaSpan) {
        const params = {
            prompt: prompt,
            max_length: parseInt(this.maxLengthInput.value) || 150,
            temperature: parseFloat(this.temperatureInput.value) || 0.8,
            learn: this.learnModeInput.checked,
        };
        
        // Use fetch with streaming for SSE
        fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params),
        }).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let generatedText = '';
            
            const processChunk = ({ done, value }) => {
                if (done) {
                    this.finishGeneration(textSpan, metaSpan);
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete SSE messages
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            this.handleSSEEvent(data, textSpan, (char) => {
                                generatedText += char;
                            });
                        } catch (e) {
                            console.error('Failed to parse SSE:', e, line);
                        }
                    }
                }
                
                reader.read().then(processChunk);
            };
            
            reader.read().then(processChunk);
        }).catch(error => {
            console.error('Generation error:', error);
            textSpan.textContent = 'Error generating response. Please try again.';
            this.finishGeneration(textSpan, metaSpan);
        });
    }
    
    handleSSEEvent(data, textSpan, onChar) {
        switch (data.type) {
            case 'phase':
                this.updatePhaseCard(data);
                break;
            
            case 'char':
                // Character-by-character animation
                textSpan.textContent += data.char;
                onChar(data.char);
                // Auto-scroll
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
                break;
            
            case 'memory':
                this.showMemoryResults(data.details?.recalled || []);
                break;
            
            case 'metric':
                console.log('Final metrics:', data.details);
                break;
            
            case 'done':
                // Will be handled by finishGeneration
                break;
            
            case 'error':
                textSpan.textContent += `\n\nError: ${data.message}`;
                break;
        }
    }
    
    updatePhaseCard(data) {
        const phaseId = `phase-${data.phase}`;
        let card = document.getElementById(phaseId);
        
        if (!card) {
            // Create new phase card
            card = document.createElement('div');
            card.id = phaseId;
            card.className = `phase-card ${data.status}`;
            card.innerHTML = `
                <div class="phase-header">
                    <span class="phase-badge phase-${data.phase}">${data.phase}</span>
                    <span class="phase-title">${data.name}</span>
                    <span class="phase-status ${data.status}">${data.status}</span>
                </div>
                <div class="phase-details"></div>
            `;
            this.phasesContainer.appendChild(card);
        } else {
            // Update existing card
            card.className = `phase-card ${data.status}`;
            const statusSpan = card.querySelector('.phase-status');
            statusSpan.className = `phase-status ${data.status}`;
            statusSpan.textContent = data.status;
        }
        
        // Update details
        if (data.details && data.status === 'complete') {
            const detailsDiv = card.querySelector('.phase-details');
            detailsDiv.innerHTML = this.formatPhaseDetails(data.details);
        }
        
        // Scroll to show latest phase
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    formatPhaseDetails(details) {
        let html = '';
        for (const [key, value] of Object.entries(details)) {
            if (key === 'recalled_facts') continue; // Skip for memory panel
            
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let displayValue = value;
            
            // Highlight certain values
            let valueClass = '';
            if (key === 'complexity' && value === 'O(log n)') {
                valueClass = 'highlight';
            } else if (key === 'forgetting_score' && value === 0) {
                valueClass = 'highlight';
                displayValue = '0.0000';
            } else if (key === 'uses_backprop' && value === false) {
                valueClass = 'highlight';
                displayValue = 'No ✓';
            } else if (key === 'global_gradients' && value === 0) {
                valueClass = 'highlight';
                displayValue = '0 ✓';
            }
            
            // Format arrays
            if (Array.isArray(value)) {
                displayValue = `[${value.join(', ')}]`;
            }
            
            html += `
                <div class="phase-detail-row">
                    <span class="phase-detail-label">${label}</span>
                    <span class="phase-detail-value ${valueClass}">${displayValue}</span>
                </div>
            `;
        }
        return html;
    }
    
    showMemoryResults(results) {
        if (!results.length) return;
        
        this.memoryPanel.style.display = 'block';
        this.memoryResults.innerHTML = results.map(r => `
            <div class="memory-item">
                <div class="fact">${r.fact}</div>
                <div class="similarity">Similarity: ${r.similarity}</div>
            </div>
        `).join('');
    }
    
    finishGeneration(textSpan, metaSpan) {
        this.isGenerating = false;
        this.thinkingStatus.textContent = 'Complete';
        this.thinkingStatus.classList.remove('active');
        this.updateSendButton();
        
        // Remove cursor
        const cursor = textSpan.querySelector('.cursor');
        if (cursor) cursor.remove();
        
        // Add timestamp
        const now = new Date();
        metaSpan.textContent = `Generated at ${now.toLocaleTimeString()}`;
    }
    
    clearThinkingPanel() {
        // Clear phase cards but keep placeholder structure
        this.phasesContainer.innerHTML = '';
        this.memoryPanel.style.display = 'none';
        this.memoryResults.innerHTML = '';
    }
    
    addMessage(text, role, isStreaming = false) {
        const msg = document.createElement('div');
        msg.className = `message ${role}`;
        
        const avatar = role === 'user' ? '👤' : '⚛';
        
        msg.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${text}${isStreaming ? '<span class="cursor"></span>' : ''}</div>
                <div class="message-meta"></div>
            </div>
        `;
        
        this.chatContainer.appendChild(msg);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        
        return msg;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.fluxChat = new FluxChat();
});
