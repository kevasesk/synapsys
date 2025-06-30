document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const dialogArea = document.getElementById('dialog-area');
    const typingIndicator = document.getElementById('typing-indicator');

    // --- Helper function to scroll to the bottom of the chat ---
    const scrollToBottom = () => {
        dialogArea.scrollTop = dialogArea.scrollHeight;
    };
    
    // --- Helper function to create and append a message bubble ---
    const appendMessage = (data, author) => {
        console.log(data);
        
        const messageWrapper = document.createElement('div');
        const messageBubble = document.createElement('div');
        const messageContent = document.createElement('div'); // Use a div for richer content (like line breaks)

        if (author === 'user') {
            messageWrapper.className = 'flex justify-end';
            messageBubble.className = 'bg-blue-600 p-3 rounded-lg max-w-xs';
            const messageText = document.createElement('p');
            messageText.className = 'text-sm';
            messageText.textContent = data;
            messageContent.appendChild(messageText);
        } else {
            // --- BOT MESSAGE LOGIC ---
            messageWrapper.className = 'flex justify-start';
            
            // Default styles for error or simple string messages
            let bgColor = ' bg-gray-700 ';
            let messageText = '';

            // Check if data is the expected object or a fallback string (e.g., error message)
            if (typeof data === 'object' && data.sentiment_data && data.spam_data) {
                const sentimentData = data.sentiment_data;
                const spamData = data.spam_data;

                // --- 1. Sentiment Analysis ---
                // Convert percentage strings to numbers for comparison
                const positiveProb = parseFloat(sentimentData.positive_proba);
                const negativeProb = parseFloat(sentimentData.negative_proba);
                
                let sentimentResult = 'Neutral'; // Default to Neutral
                
                // Set a threshold for neutrality. If the difference in probabilities is less than 20%, we'll call it neutral.
                const NEUTRAL_THRESHOLD = 20; 

                if (Math.abs(positiveProb - negativeProb) < NEUTRAL_THRESHOLD) {
                    sentimentResult = 'Neutral';
                } else if (positiveProb > negativeProb) {
                    sentimentResult = 'Positive';
                } else {
                    sentimentResult = 'Negative';
                }
                
                const sentimentLine = `<b>Sentiment:</b> ${sentimentResult} <br><small>(Positive: ${sentimentData.positive_proba}, Negative: ${sentimentData.negative_proba})</small>`;

                // --- 2. Spam Analysis ---
                const isSpam = spamData.spam_proba > spamData.not_spam_proba;
                const spamResult = isSpam ? 'Spam' : 'Not Spam';
                const spamLine = `<b>Detection:</b> ${spamResult} <br><small>(Spam: ${spamData.spam_proba}, Not Spam: ${spamData.not_spam_proba})</small>`;
                
                // --- 3. Combine messages ---
                messageText = `${sentimentLine}<hr class="my-2 border-gray-500">${spamLine}`;

            } else {
                // Fallback for simple string messages like errors
                messageText = data.toString();
            }
            
            messageBubble.className = `bg-gray-700 p-3 rounded-lg max-w-xs`;
            messageContent.className = 'text-sm';
            messageContent.innerHTML = messageText;
        }

        messageBubble.appendChild(messageContent);
        messageWrapper.appendChild(messageBubble);
        dialogArea.appendChild(messageWrapper);
        scrollToBottom();
    };

    // --- Handle form submission ---
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent page reload
        
        const messageText = userInput.value.trim();
        if (!messageText) {
            return; // Do nothing if input is empty
        }

        // 1. Display user's message immediately
        appendMessage(messageText, 'user');
        userInput.value = ''; // Clear the input field

        // 2. Show typing indicator
        typingIndicator.classList.remove('hidden');
        scrollToBottom();

        try {
            // 3. Send message to Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            typingIndicator.classList.add('hidden');
            appendMessage(data, 'bot');

        } catch (error) {
            console.error('Fetch error:', error);
            typingIndicator.classList.add('hidden');
            appendMessage('Sorry, something went wrong. Please try again.', 'bot');
        }
    });

    scrollToBottom();
});