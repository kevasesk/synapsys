window.Chat = function () {
    return {
        messages: [
            { type: 'bot', text: 'Welcome! Choose a film to write a comment below. The bot will analyze this comment.' }
        ],
        messageInput: '',
        isTyping: false,
        selectedMode: 'sentiment',
        uploadedFile: null,

        setMode: function(mode) {
            this.selectedMode = mode;
        },

        handleFileUpload: async function(event) {
            const file = event.target.files[0];
            if (!file) return;

            const allowedTypes = ['text/plain', 'application/pdf', 'text/csv'];
            if (!allowedTypes.includes(file.type)) {
                alert('Invalid file type. Please upload only TXT, PDF, or CSV files.');
                return;
            }

            if (file.size > 20 * 1024 * 1024) { // 20 MB
                alert('File is too large. Maximum size is 20 MB.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    this.uploadedFile = { name: result.filename };
                } else {
                    alert('File upload failed.');
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('An error occurred during upload.');
            }
        },

        removeFile: async function() {
            if (!this.uploadedFile) return;

            try {
                const response = await fetch('/delete_file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: this.uploadedFile.name })
                });

                if (response.ok) {
                    this.uploadedFile = null;
                    document.getElementById('file-upload').value = '';
                } else {
                    alert('Failed to delete file from server.');
                }
            } catch (error) {
                console.error('Delete error:', error);
                alert('An error occurred while deleting the file.');
            }
        },

        sendMessage: async function() {
            const userMessage = this.messageInput.trim();
            if (!userMessage && !this.uploadedFile) return;

            this.messages.push({
                type: 'user',
                text: userMessage
            });

            this.messageInput = '';
            this.$nextTick(() => this.scrollToBottom());
            this.isTyping = true;

            try {
                const params = new URLSearchParams({
                    message: userMessage,
                    mode: this.selectedMode
                });
                if (this.uploadedFile) {
                    params.append('filename', this.uploadedFile.name);
                }

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: params.toString()
                });
                const botResponse = await response.text();
                this.messages.push({
                    type: 'bot',
                    text: botResponse
                });

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