/* JavaScript utilities for the web app */

// Smooth scrolling to results
function scrollToResults() {
    const resultsElement = document.getElementById('results');
    if (resultsElement && resultsElement.style.display !== 'none') {
        resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Format percentage for display
function formatPercent(value) {
    if (typeof value === 'number') {
        return (value * 100).toFixed(1) + '%';
    }
    return value;
}

// Validate age input
function validateAge(age) {
    return age >= 1 && age <= 120;
}

// Validate numerical ranges
function validateRange(value, min, max) {
    return value >= min && value <= max;
}

// Show loading state
function showLoading() {
    const btn = document.querySelector('button[type="submit"]');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Assessing Risk...';
    }
}

// Hide loading state
function hideLoading() {
    const btn = document.querySelector('button[type="submit"]');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = 'Assess Risk';
    }
}

// Initialize form with event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Toggle cigarettes per day based on smoker status
    const smokerSelect = document.getElementById('currentSmoker');
    const cigsInput = document.getElementById('cigsPerDay');
    
    if (smokerSelect && cigsInput) {
        smokerSelect.addEventListener('change', function() {
            cigsInput.disabled = (this.value === '0');
            if (this.value === '0') {
                cigsInput.value = '0';
            }
        });
    }

    // Form submission
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            showLoading();

            setTimeout(() => {
                hideLoading();
                scrollToResults();
            }, 1500);
        });
    }

    // Copy result to clipboard functionality (optional enhancement)
    const resultElements = document.querySelectorAll('[data-copy]');
    resultElements.forEach(el => {
        el.addEventListener('click', function() {
            const text = this.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const original = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = original;
                }, 2000);
            });
        });
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }

    // Escape to dismiss alerts
    if (e.key === 'Escape') {
        const alerts = document.querySelectorAll('.alert-dismissible');
        alerts.forEach(alert => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) closeBtn.click();
        });
    }
});

// Log application version
console.log('Cardiovascular Risk Predictor - v1.0');
console.log('Educational Purpose Only - Not for Clinical Use');
