/* =========================================================
   Dengue Prediction System — Main JavaScript
   ========================================================= */

'use strict';

/* =========================================================
   SIDEBAR TOGGLE
   ========================================================= */

(function initSidebar() {
  const sidebar = document.getElementById('appSidebar');
  const mainContent = document.getElementById('mainContent');
  const topbar = document.getElementById('appTopbar');
  const footer = document.getElementById('appFooter');
  const collapseBtn = document.getElementById('sidebarCollapseBtn');
  const hamburgerBtn = document.getElementById('sidebarHamburger');
  const COLLAPSED_KEY = 'sidebarCollapsed';

  if (!sidebar) return;

  // Restore collapsed state
  if (localStorage.getItem(COLLAPSED_KEY) === 'true') {
    sidebar.classList.add('collapsed');
    topbar && topbar.classList.add('sidebar-collapsed');
    mainContent && mainContent.classList.add('sidebar-collapsed');
    footer && footer.classList.add('sidebar-collapsed');
    updateCollapseIcon(true);
  }

  // Desktop collapse button
  if (collapseBtn) {
    collapseBtn.addEventListener('click', function () {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      topbar && topbar.classList.toggle('sidebar-collapsed', isCollapsed);
      mainContent && mainContent.classList.toggle('sidebar-collapsed', isCollapsed);
      footer && footer.classList.toggle('sidebar-collapsed', isCollapsed);
      localStorage.setItem(COLLAPSED_KEY, isCollapsed);
      updateCollapseIcon(isCollapsed);
    });
  }

  // Mobile hamburger
  if (hamburgerBtn) {
    hamburgerBtn.addEventListener('click', function () {
      sidebar.classList.toggle('mobile-open');
    });
  }

  // Close sidebar on overlay click (mobile)
  document.addEventListener('click', function (e) {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('mobile-open') &&
        !sidebar.contains(e.target) &&
        e.target !== hamburgerBtn) {
      sidebar.classList.remove('mobile-open');
    }
  });

  function updateCollapseIcon(collapsed) {
    const icon = collapseBtn && collapseBtn.querySelector('i');
    if (icon) {
      icon.className = collapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
    }
  }

  // Submenu toggles
  document.querySelectorAll('[data-submenu]').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      if (sidebar.classList.contains('collapsed')) return;
      e.preventDefault();
      const targetId = btn.getAttribute('data-submenu');
      const submenu = document.getElementById(targetId);
      if (!submenu) return;
      const isOpen = submenu.classList.toggle('open');
      btn.classList.toggle('submenu-open', isOpen);
    });
  });

  // Auto-open submenu that contains active link
  document.querySelectorAll('.nav-submenu').forEach(function (submenu) {
    if (submenu.querySelector('.nav-link-item.active')) {
      submenu.classList.add('open');
      const toggleBtn = document.querySelector(`[data-submenu="${submenu.id}"]`);
      toggleBtn && toggleBtn.classList.add('submenu-open');
    }
  });
})();


/* =========================================================
   ACTIVE NAV LINK DETECTION
   ========================================================= */

(function markActiveNavLinks() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link-item[href]').forEach(function (link) {
    const href = link.getAttribute('href');
    if (href && href !== '/' && path.startsWith(href)) {
      link.classList.add('active');
    } else if (href === '/' && path === '/') {
      link.classList.add('active');
    }
  });
})();

/* =========================================================
   PUBLIC MOBILE MENU
   ========================================================= */
(function initPublicMobileMenu() {
  const btn  = document.getElementById('publicNavHamburger');
  const menu = document.getElementById('publicMobileMenu');
  if (!btn || !menu) return;

  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    const isOpen = menu.classList.toggle('open');
    btn.querySelector('i').className = isOpen ? 'fas fa-times' : 'fas fa-bars';
  });

  document.addEventListener('click', function (e) {
    if (!menu.contains(e.target) && e.target !== btn) {
      menu.classList.remove('open');
      btn.querySelector('i').className = 'fas fa-bars';
    }
  });
}());

/* =========================================================
   TOAST NOTIFICATION SYSTEM
   ========================================================= */

const Toast = (function () {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = {
    success: 'fas fa-check-circle',
    danger: 'fas fa-exclamation-circle',
    warning: 'fas fa-exclamation-triangle',
    info: 'fas fa-info-circle',
  };

  function show(message, type = 'info', title = '', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const iconClass = icons[type] || icons.info;
    const titleHtml = title ? `<div class="toast-title">${title}</div>` : '';

    toast.innerHTML = `
      <i class="toast-icon ${iconClass}"></i>
      <div class="toast-content">
        ${titleHtml}
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close" aria-label="Close"><i class="fas fa-times"></i></button>
      <div class="toast-progress"></div>
    `;

    toast.querySelector('.toast-close').addEventListener('click', function () {
      dismiss(toast);
    });

    container.appendChild(toast);

    if (duration > 0) {
      setTimeout(function () { dismiss(toast); }, duration);
    }

    return toast;
  }

  function dismiss(toast) {
    if (!toast || !toast.parentNode) return;
    toast.classList.add('hiding');
    toast.addEventListener('animationend', function () {
      toast.parentNode && toast.parentNode.removeChild(toast);
    }, { once: true });
  }

  return { show, dismiss };
})();

// Expose globally
window.showToast = function (message, type, title, duration) {
  return Toast.show(message, type, title, duration);
};

// Convert flash messages into toasts on load
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('[data-flash]').forEach(function (el) {
    const type = el.getAttribute('data-flash-type') || 'info';
    const msg  = el.getAttribute('data-flash');
    if (msg) Toast.show(msg, type);
    el.remove();
  });
});


/* =========================================================
   LOADING OVERLAY
   ========================================================= */

const Loader = (function () {
  let overlay = null;

  function show(message = 'Loading...', detail = 'Please wait') {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'loading-overlay';
      overlay.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-text" id="loaderMessage">${message}</div>
        <div style="font-size:12px;color:#94a3b8" id="loaderDetail">${detail}</div>
      `;
      document.body.appendChild(overlay);
    } else {
      document.getElementById('loaderMessage').textContent = message;
      document.getElementById('loaderDetail').textContent = detail;
    }
  }

  function hide() {
    if (overlay && overlay.parentNode) {
      overlay.parentNode.removeChild(overlay);
      overlay = null;
    }
  }

  return { show, hide };
})();

window.showLoader = Loader.show;
window.hideLoader = Loader.hide;


/* =========================================================
   AJAX HELPERS
   ========================================================= */

async function apiGet(url) {
  try {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('GET error:', url, err);
    throw err;
  }
}

async function apiPost(url, data) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('POST error:', url, err);
    throw err;
  }
}

async function apiDelete(url) {
  try {
    const res = await fetch(url, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('DELETE error:', url, err);
    throw err;
  }
}

window.apiGet    = apiGet;
window.apiPost   = apiPost;
window.apiDelete = apiDelete;


/* =========================================================
   TABLE SEARCH / FILTER UTILITY
   ========================================================= */

window.initTableSearch = function (inputId, tableId) {
  const input = document.getElementById(inputId);
  const table = document.getElementById(tableId);
  if (!input || !table) return;

  input.addEventListener('input', function () {
    const query = this.value.toLowerCase().trim();
    const rows  = table.querySelectorAll('tbody tr');
    let visible = 0;
    rows.forEach(function (row) {
      const text = row.textContent.toLowerCase();
      const show = !query || text.includes(query);
      row.style.display = show ? '' : 'none';
      if (show) visible++;
    });

    // Handle empty state
    let emptyRow = table.querySelector('.table-empty-search');
    if (visible === 0 && !emptyRow) {
      emptyRow = document.createElement('tr');
      emptyRow.className = 'table-empty-search';
      const cols = table.querySelectorAll('thead th').length;
      emptyRow.innerHTML = `<td colspan="${cols}" class="table-empty">
        <i class="fas fa-search"></i>No results for "<strong>${query}</strong>"
      </td>`;
      table.querySelector('tbody').appendChild(emptyRow);
    } else if (visible > 0 && emptyRow) {
      emptyRow.remove();
    }
  });
};


/* =========================================================
   CUSTOM TABS (lightweight)
   ========================================================= */

(function initTabs() {
  document.querySelectorAll('[data-tab-target]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const targetId = btn.getAttribute('data-tab-target');
      const container = btn.closest('[data-tabs]') || btn.parentElement.parentElement;

      // Deactivate all tabs/panels in container
      container.querySelectorAll('[data-tab-target]').forEach(function (b) {
        b.classList.remove('active');
      });
      document.querySelectorAll('.tab-panel').forEach(function (p) {
        p.classList.remove('active');
      });

      btn.classList.add('active');
      const target = document.getElementById(targetId);
      target && target.classList.add('active');
    });
  });
})();


/* =========================================================
   DATE & NUMBER HELPERS
   ========================================================= */

window.formatDate = function (dateStr) {
  if (!dateStr) return 'N/A';
  const d = new Date(dateStr);
  return d.toLocaleDateString('id-ID', { year: 'numeric', month: 'short', day: 'numeric' });
};

window.formatNumber = function (n) {
  if (n === null || n === undefined) return '0';
  return Number(n).toLocaleString('id-ID');
};

window.monthName = function (m) {
  const names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return names[(parseInt(m) - 1)] || m;
};


/* =========================================================
   COUNTER ANIMATION
   ========================================================= */

window.animateCounter = function (el, target, duration = 1200) {
  const start = 0;
  const startTime = performance.now();
  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(start + (target - start) * eased).toLocaleString('id-ID');
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
};

// Auto-animate elements with data-counter
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('[data-counter]').forEach(function (el) {
    const target = parseInt(el.getAttribute('data-counter'));
    if (!isNaN(target)) animateCounter(el, target);
  });
});


/* =========================================================
   CHART DEFAULT CONFIGURATION
   ========================================================= */

if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size   = 12;
  Chart.defaults.color       = '#64748b';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.backgroundColor = '#1e293b';
  Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
  Chart.defaults.plugins.tooltip.bodyColor  = '#94a3b8';
  Chart.defaults.plugins.tooltip.borderColor = '#334155';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.scale.grid.color = '#f1f5f9';
  Chart.defaults.scale.border = { display: false };
}

// Color palette for charts
window.chartColors = {
  primary:   '#0ea5e9',
  secondary: '#8b5cf6',
  success:   '#10b981',
  warning:   '#f59e0b',
  danger:    '#ef4444',
  orange:    '#f97316',
  palette:   ['#0ea5e9','#8b5cf6','#10b981','#f59e0b','#ef4444','#f97316'],
  alphas: {
    primary:   'rgba(14,165,233,0.15)',
    secondary: 'rgba(139,92,246,0.15)',
    success:   'rgba(16,185,129,0.15)',
    warning:   'rgba(245,158,11,0.15)',
    danger:    'rgba(239,68,68,0.15)',
  }
};

// Helper: create smooth line dataset
window.lineDataset = function (label, data, color) {
  color = color || window.chartColors.primary;
  return {
    label: label,
    data: data,
    borderColor: color,
    backgroundColor: color.replace(')', ',0.12)').replace('rgb', 'rgba'),
    borderWidth: 2.5,
    tension: 0.4,
    fill: true,
    pointBackgroundColor: color,
    pointRadius: 4,
    pointHoverRadius: 6,
  };
};


/* =========================================================
   AUTO-DISMISS BOOTSTRAP ALERTS (fallback)
   ========================================================= */

document.addEventListener('DOMContentLoaded', function () {
  setTimeout(function () {
    document.querySelectorAll('.alert-dismissible[data-auto-dismiss]').forEach(function (el) {
      el.style.transition = 'opacity 0.5s';
      el.style.opacity = '0';
      setTimeout(function () { el.remove(); }, 500);
    });
  }, 5000);
});


/* =========================================================
   CONFIRM DELETE HELPERS
   ========================================================= */

window.confirmAction = function (message, onConfirm) {
  // Simple modal-based confirm (fallback to browser confirm)
  if (window.confirm(message)) {
    onConfirm();
  }
};


/* =========================================================
   BOOTSTRAP MODAL WRAPPERS (compatibility)
   ========================================================= */

window.showModal = function (id) {
  const el = document.getElementById(id);
  if (!el) return;
  if (typeof bootstrap !== 'undefined') {
    const m = bootstrap.Modal.getOrCreateInstance(el);
    m.show();
  }
};

window.hideModal = function (id) {
  const el = document.getElementById(id);
  if (!el) return;
  if (typeof bootstrap !== 'undefined') {
    const m = bootstrap.Modal.getInstance(el);
    m && m.hide();
  }
};


/* =========================================================
   FORM LOADING STATE
   ========================================================= */

window.setButtonLoading = function (btn, loading, text) {
  if (!btn) return;
  if (loading) {
    btn.setAttribute('data-original', btn.innerHTML);
    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>${text || 'Processing...'}`;
    btn.disabled = true;
  } else {
    btn.innerHTML = btn.getAttribute('data-original') || btn.innerHTML;
    btn.disabled = false;
  }
};


/* =========================================================
   INIT ON DOM READY
   ========================================================= */

document.addEventListener('DOMContentLoaded', function () {
  // Activate first tab if present
  document.querySelectorAll('[data-tabs]').forEach(function (tabGroup) {
    const firstBtn = tabGroup.querySelector('[data-tab-target]');
    if (firstBtn && !tabGroup.querySelector('[data-tab-target].active')) {
      firstBtn.click();
    }
  });
});
