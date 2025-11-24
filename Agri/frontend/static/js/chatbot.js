/**
 * AgriIntel Chatbot JavaScript
 * Handles chat functionality and n8n webhook integration
 */

const WEBHOOK_URL = 'https://projectu.app.n8n.cloud/webhook/agri-intel-chat';

// Chat state
let chatHistory = [];
let isWaitingForResponse = false;

// DOM Elements
let chatMessages;
let messageInput;
let sendButton;
let typingIndicator;
let chatContainer;
let chatStatus;

// Initialize chat
document.addEventListener('DOMContentLoaded', () => {
    console.log('Chatbot initializing...');
    
    // Get DOM elements
    chatMessages = document.getElementById('chatMessages');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    typingIndicator = document.getElementById('typingIndicator');
    chatContainer = document.getElementById('chatContainer');
    chatStatus = document.getElementById('chatStatus');
    
    // Verify all DOM elements are found
    if (!chatMessages) console.error('chatMessages element not found');
    if (!messageInput) console.error('messageInput element not found');
    if (!sendButton) console.error('sendButton element not found');
    if (!typingIndicator) console.error('typingIndicator element not found');
    
    if (messageInput) {
        // Auto-resize textarea
        messageInput.addEventListener('input', autoResizeTextarea);
        
        // Send message on Enter (Shift+Enter for new line)
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Test webhook connection
    testWebhookConnection();
    
    console.log('Chatbot initialized successfully');
});

/**
 * Test webhook connection on load
 */
async function testWebhookConnection() {
    try {
        console.log('Testing webhook connection to:', WEBHOOK_URL);
        const response = await fetch(WEBHOOK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: 'test',
                sessionId: 'test-connection'
            })
        });
        
        if (response.ok) {
            console.log('Webhook connection successful');
            if (chatStatus) chatStatus.textContent = 'Online';
        } else {
            console.warn('Webhook returned status:', response.status);
            if (chatStatus) chatStatus.textContent = 'Limited connectivity';
        }
    } catch (error) {
        console.error('Webhook connection test failed:', error);
        if (chatStatus) chatStatus.textContent = 'Offline - Check connection';
    }
}

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

/**
 * Toggle chat window visibility
 */
function toggleChat() {
    chatContainer.classList.toggle('minimized');
    floatingBtn.classList.toggle('show');
    
    if (!chatContainer.classList.contains('minimized')) {
        messageInput.focus();
        notificationBadge.classList.remove('show');
    }
}

/**
 * Set predefined message from hint chips
 */
function setMessage(message) {
    messageInput.value = message;
    autoResizeTextarea();
    messageInput.focus();
}

/**
 * Send message to webhook
 */
async function sendMessage() {
    const message = messageInput.value.trim();
    
    // Validate input
    if (!message || isWaitingForResponse) {
        return;
    }

    // Clear input
    messageInput.value = '';
    autoResizeTextarea();
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Store in history
    chatHistory.push({
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
    });

    // Disable input while waiting
    isWaitingForResponse = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    
    // Show typing indicator
    showTypingIndicator();

    try {
        // Send to n8n webhook
        const response = await fetch(WEBHOOK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-5), // Send last 5 messages for context
                timestamp: new Date().toISOString(),
                sessionId: getOrCreateSessionId()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Extract response from webhook
        let botMessage = data.response || data.message || data.output || data.text;
        
        // Handle different response formats
        if (typeof botMessage === 'object') {
            botMessage = JSON.stringify(botMessage, null, 2);
        }
        
        if (!botMessage) {
            botMessage = "I apologize, but I didn't receive a proper response. Could you please try asking your question again?";
        }

        // Add bot response to chat
        addMessage(botMessage, 'bot');
        
        // Store in history
        chatHistory.push({
            role: 'assistant',
            content: botMessage,
            timestamp: new Date().toISOString()
        });

        // Show notification if chat is minimized
        if (chatContainer.classList.contains('minimized')) {
            notificationBadge.classList.add('show');
        }

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        
        // Show detailed error message
        let errorMsg = `I'm having trouble connecting to the AI service. `;
        
        if (error.message === 'Failed to fetch') {
            errorMsg += `The n8n webhook at ${WEBHOOK_URL} is not responding. Please verify:\n`;
            errorMsg += `1. The webhook URL is correct\n`;
            errorMsg += `2. Your n8n workflow is active\n`;
            errorMsg += `3. CORS is enabled on the webhook\n`;
            errorMsg += `4. Your internet connection is working`;
        } else {
            errorMsg += `Error: ${error.message}`;
        }
        
        addErrorMessage(errorMsg);
        
        // Update status
        if (chatStatus) chatStatus.textContent = 'Connection error';
    } finally {
        // Re-enable input
        isWaitingForResponse = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
}

/**
 * Add message to chat
 */
function addMessage(text, sender = 'bot') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timestamp = formatTimestamp(new Date());
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                ${sender === 'bot' 
                    ? '<path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>'
                    : '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>'
                }
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                ${formatMessageText(text)}
            </div>
            <div class="message-time">${timestamp}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add error message to chat
 */
function addErrorMessage(text) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style="width: 20px; height: 20px;">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <span>${text}</span>
    `;
    
    chatMessages.appendChild(errorDiv);
    scrollToBottom();
}

/**
 * Format message text (preserve line breaks, links, etc.)
 */
function formatMessageText(text) {
    // Convert markdown-style links
    text = text.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Convert URLs to links
    text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    
    // Convert bold text
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert italic text
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    return text;
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    typingIndicator.classList.add('show');
    scrollToBottom();
}

/**
 * Hide typing indicator
 */
function hideTypingIndicator() {
    typingIndicator.classList.remove('show');
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
}

/**
 * Format timestamp
 */
function formatTimestamp(date) {
    const now = new Date();
    const diff = Math.floor((now - date) / 1000); // difference in seconds
    
    if (diff < 60) {
        return 'Just now';
    } else if (diff < 3600) {
        const minutes = Math.floor(diff / 60);
        return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (diff < 86400) {
        const hours = Math.floor(diff / 3600);
        return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
        return date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    }
}

/**
 * Get or create session ID
 */
function getOrCreateSessionId() {
    let sessionId = localStorage.getItem('agriintel_session_id');
    
    if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('agriintel_session_id', sessionId);
    }
    
    return sessionId;
}

/**
 * Clear chat history
 */
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        chatHistory = [];
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <p>👋 Chat history cleared! How can I assist you today?</p>
                    </div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
        `;
        
        // Clear session
        localStorage.removeItem('agriintel_session_id');
    }
}

/**
 * Export chat history
 */
function exportChat() {
    const chatText = chatHistory.map(msg => 
        `[${msg.role.toUpperCase()}] ${new Date(msg.timestamp).toLocaleString()}\n${msg.content}\n`
    ).join('\n---\n\n');
    
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agriintel_chat_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Handle connection status
window.addEventListener('online', () => {
    console.log('Connection restored');
});

window.addEventListener('offline', () => {
    addErrorMessage('You are currently offline. Please check your internet connection.');
});
