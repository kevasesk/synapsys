window.Chat = function () {
    return {
        messages: [
            { type: 'bot', text: 'Welcome! Choose a film to write a comment below. The bot will analyze this comment.' }
        ],
        messageInput: '',
        isTyping: false,

        sendMessage: async function() {
            const userMessage = this.messageInput.trim();
            if (!userMessage) return;

            this.messages.push({ type: 'user', text: userMessage });
            this.messageInput = '';
            this.$nextTick(() => this.scrollToBottom());

            this.isTyping = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ message: userMessage }).toString()
                });
                const botResponse = await response.text();
                this.messages.push({ type: 'bot', text: botResponse });

            } finally {
                this.isTyping = false;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        scrollToBottom: function() {
            const dialogArea = this.$refs.dialogArea;
            dialogArea.scrollTop = dialogArea.scrollHeight;
        }
    };
};