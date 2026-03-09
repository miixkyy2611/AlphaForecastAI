
(() => {
  const storageKey = 'alphaforecast-theme';
  const body = document.body;
  const buttons = document.querySelectorAll('[data-theme-choice]');
  const saved = localStorage.getItem(storageKey) || 'aurora';
  body.dataset.theme = saved;

  const syncButtons = () => {
    buttons.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.themeChoice === body.dataset.theme);
    });
  };

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      body.dataset.theme = btn.dataset.themeChoice;
      localStorage.setItem(storageKey, body.dataset.theme);
      syncButtons();
    });
  });

  syncButtons();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll('[data-reveal]').forEach((el) => observer.observe(el));
})();
