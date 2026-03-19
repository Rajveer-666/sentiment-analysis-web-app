document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('sentimentForm');
    const textInput = document.getElementById('textInput');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    // Set the API URL - change this to your deployed backend URL
    const API_URL = 'https://your-app-name.onrender.com';

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const text = textInput.value.trim();
        
        if (!text) {
            showResult('Please enter some text to analyze.', 'error');
            return;
        }
        
        // Show loading, hide previous results
        loadingDiv.classList.add('show');
        resultDiv.classList.remove('show', 'positive', 'negative', 'neutral', 'error');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing...';
        
        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });
            
            if (!response.ok) {
                throw new Error('Server responded with ' + response.status);
            }
            
            const data = await response.json();
            
            loadingDiv.classList.remove('show');
            
            if (data.sentiment && data.confidence) {
                showResult(data.sentiment, data.sentiment.toLowerCase(), data.confidence);
            } else {
                showResult('Could not analyze sentiment. Please try again.', 'error');
            }
            
        } catch (error) {
            console.error('Error:', error);
            loadingDiv.classList.remove('show');
            showResult('Failed to connect to the server. Please try again later.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Sentiment';
        }
    });
    
    function showResult(sentiment, className, confidence) {
        resultDiv.className = 'result show ' + className;
        resultDiv.innerHTML = '<h3>' + sentiment + '</h3>' +
            (confidence !== undefined ? '<p class="confidence">Confidence: ' + confidence.toFixed(2) + '%</p>' : '');
    }
    
    // Clear result when user starts typing
    textInput.addEventListener('input', function() {
        resultDiv.classList.remove('show', 'positive', 'negative', 'neutral', 'error');
    });
});
