/* Shared front-end helpers: CSRF-aware API client, toasts, loader, theme, language, charts, safe HTML. */
(function () {
  const App = window.App = {};
  const csrf = () => (document.querySelector('meta[name="csrf-token"]') || {}).content || '';
  App.esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  App.$ = (s, r) => (r || document).querySelector(s);
  App.$$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  App.lang = () => localStorage.getItem('lang') || 'en';
  App.t = k => (I18N[App.lang()] || I18N.en)[k] || I18N.en[k] || k;

  App.toast = function (msg, type) {
    const w = App.$('.toast-wrap'); if (!w) return alert(msg);
    const d = document.createElement('div'); d.className = 'tst ' + (type || 'info'); d.textContent = msg; w.appendChild(d);
    setTimeout(() => { d.style.opacity = 0; d.style.transition = 'opacity .4s'; setTimeout(() => d.remove(), 450); }, type === 'danger' ? 7000 : 4000);
  };
  let loads = 0;
  App.loading = on => { loads = Math.max(0, loads + (on ? 1 : -1)); const l = App.$('#loader'); if (l) l.classList.toggle('on', loads > 0); };
  App.api = async function (url, o) {
    o = o || {}; const opt = {method: o.method || 'GET', headers: {'X-CSRFToken': csrf(), 'Accept': 'application/json'}, credentials: 'same-origin'};
    if (o.form) opt.body = o.form; else if (o.body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(o.body); }
    if (!o.quiet) App.loading(true);
    try {
      const r = await fetch(url, opt); let j = null; try { j = await r.json(); } catch (e) {}
      if (!r.ok) { const e = new Error((j && j.error) || ('Request failed (' + r.status + ')')); e.status = r.status; e.data = j; throw e; }
      return j;
    } finally { if (!o.quiet) App.loading(false); }
  };
  App.fail = e => App.toast(e.message || 'Something went wrong', 'danger');

  /* theme */
  App.setTheme = t => { document.documentElement.setAttribute('data-bs-theme', t); localStorage.setItem('theme', t); const b = App.$('#themeBtn i'); if (b) b.className = t === 'dark' ? 'bi bi-sun' : 'bi bi-moon-stars'; App.recolor(); };
  App.toggleTheme = () => App.setTheme(document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark');
  /* language */
  App.applyLang = function () {
    const L = App.lang(); document.documentElement.lang = L;
    App.$$('[data-i18n]').forEach(e => e.textContent = App.t(e.dataset.i18n));
    App.$$('[data-i18n-ph]').forEach(e => e.placeholder = App.t(e.dataset.i18nPh));
    const b = App.$('#langBtn'); if (b) b.textContent = L === 'en' ? 'हिं' : 'EN';
    document.dispatchEvent(new CustomEvent('langchange', {detail: L}));
  };
  App.toggleLang = () => { localStorage.setItem('lang', App.lang() === 'en' ? 'hi' : 'en'); App.applyLang(); };

  /* charts */
  const charts = {};
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  App.palette = ['#1565c0', '#ef6c00', '#2e7d32', '#8e24aa', '#c62828', '#00838f'];
  App.chart = function (id, cfg) {
    const el = typeof id === 'string' ? document.getElementById(id) : id; if (!el || !window.Chart) return null;
    if (charts[el.id]) charts[el.id].destroy();
    cfg.options = Object.assign({responsive: true, maintainAspectRatio: false, animation: {duration: 700}}, cfg.options || {});
    charts[el.id] = new Chart(el, cfg); App.styleChart(charts[el.id]); return charts[el.id];
  };
  App.styleChart = function (c) {
    const ink = css('--muted'), line = css('--line'); const o = c.options;
    if (o.scales) Object.values(o.scales).forEach(s => { s.ticks = Object.assign(s.ticks || {}, {color: ink}); s.grid = Object.assign(s.grid || {}, {color: line}); });
    o.plugins = o.plugins || {}; o.plugins.legend = Object.assign(o.plugins.legend || {}, {labels: {color: ink}}); c.update('none');
  };
  App.recolor = () => Object.values(charts).forEach(App.styleChart);
  App.fmtDate = s => { try { return new Date(s).toLocaleDateString(undefined, {day: '2-digit', month: 'short', year: 'numeric'}); } catch (e) { return s; } };
  App.badge = lvl => `<span class="badge badge-${String(lvl).toLowerCase()}">${App.esc(lvl)}</span>`;

  document.addEventListener('DOMContentLoaded', function () {
    App.setTheme(localStorage.getItem('theme') || 'light'); App.applyLang();
    const fd = App.$('#flash-data'); if (fd) JSON.parse(fd.dataset.messages || '[]').forEach(m => App.toast(m[1], m[0]));
    const mb = App.$('#menuBtn'); if (mb) mb.onclick = () => App.$('.sidebar').classList.toggle('open');
    App.$$('[data-confirm]').forEach(f => f.addEventListener('submit', e => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
  });
})();
