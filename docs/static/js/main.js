document.addEventListener('DOMContentLoaded', () => {
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    initScrollReveal();
    initNavigation();
});

function initScrollReveal() {
    const reveals = document.querySelectorAll('.reveal');
    if (!reveals.length) return;

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                revealObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    reveals.forEach(reveal => revealObserver.observe(reveal));
}

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);

            if (targetSection) {
                const navHeight = document.querySelector('nav').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight - 20;
                window.scrollTo({ top: targetPosition, behavior: 'smooth' });
            }
        });
    });

    let ticking = false;

    function updateActiveNav() {
        const scrollPos = window.scrollY + 150;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });

        ticking = false;
    }

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(updateActiveNav);
            ticking = true;
        }
    }, { passive: true });
}

function copyBibtex() {
    const bibtexContent = document.getElementById('bibtex-content');
    if (!bibtexContent) return;

    const text = bibtexContent.textContent;
    const copyBtn = document.querySelector('.copy-btn');
    const copyText = document.getElementById('copy-text');

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showCopiedFeedback(copyBtn, copyText);
        }).catch(() => fallbackCopy(text, copyBtn, copyText));
    } else {
        fallbackCopy(text, copyBtn, copyText);
    }
}

function fallbackCopy(text, btn, textSpan) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';
    document.body.appendChild(textArea);

    try {
        textArea.select();
        if (document.execCommand('copy')) {
            showCopiedFeedback(btn, textSpan);
        }
    } finally {
        document.body.removeChild(textArea);
    }
}

function showCopiedFeedback(btn, textSpan) {
    if (!btn || !textSpan) return;

    btn.classList.add('copied');
    textSpan.textContent = 'Copied!';

    setTimeout(() => {
        btn.classList.remove('copied');
        textSpan.textContent = 'Copy';
    }, 2000);
}

window.copyBibtex = copyBibtex;
