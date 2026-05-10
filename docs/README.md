# AlphaQ Project Page

This directory is a static GitHub Pages site for AlphaQ.

The page follows the `pUmpKin-Co/SparsityAndCoD/docs` template structure:

- `index.html`: page content.
- `static/css/style.css`: adapted clean academic theme.
- `static/js/main.js`: navigation, reveal animation, and BibTeX copy behavior.
- `static/images/`: paper figures exported from `AlphaQ_writing_repo`.

Deploy options:

- Use the included GitHub Actions workflow in `.github/workflows/pages.yml`.
- Or configure GitHub Pages to serve from the `docs/` folder on `main`.

Local preview:

```bash
python3 -m http.server 8000 -d docs
```

Then open `http://localhost:8000`.
