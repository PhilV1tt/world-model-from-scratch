# Setup — world-model-from-scratch

## Activation du venv

```bash
cd /Users/vitt/code/world-model-from-scratch
source .venv/bin/activate
```

Python 3.14.4. Tout passe par `./.venv/bin/python` ou `./.venv/bin/pip`. Rien en global.

## Kernel à sélectionner dans VS Code

Ouvrir un `.ipynb`, cliquer sur le sélecteur de kernel en haut à droite et choisir :

**`Python 3.14 (world-model)`** (id interne : `world-model-venv`)

Ce kernel pointe sur le binaire absolu `/Users/vitt/code/world-model-from-scratch/.venv/bin/python`, ce qui garantit que `nbformat`, `plotly`, etc. installés dans le venv sont visibles.

> Le kernel `Python 3 (ipykernel)` par défaut utilise `python` (sans chemin absolu) et part chercher l'interpréteur sur le PATH — c'est ce qui causait le `ValueError: Mime type rendering requires nbformat>=4.2.0`.

## Réinstaller le kernel

```bash
./.venv/bin/python -m ipykernel install --user \
  --name world-model-venv \
  --display-name "Python 3.14 (world-model)"
```

## Réinstaller les deps si besoin

```bash
./.venv/bin/python -m pip install --upgrade \
  "nbformat>=5.0" ipykernel plotly numpy scikit-learn nbconvert
```

## Troubleshooting Plotly

1. **`fig.show()` lance `ValueError: Mime type rendering requires nbformat>=4.2.0`**
   → Mauvais kernel sélectionné. Vérifier en haut à droite du notebook qu'il est bien sur `Python 3.14 (world-model)`.

2. **Le kernel `Python 3.14 (world-model)` n'apparaît pas**
   → Réenregistrer (commande ci-dessus), puis dans VS Code : `Cmd+Shift+P` → `Jupyter: Select Interpreter to start Jupyter server` → choisir `./.venv/bin/python`.

3. **Toujours rien après changement de kernel**
   → Restart du kernel (`Cmd+Shift+P` → `Jupyter: Restart Kernel`), puis re-exécuter la cellule.

4. **Vérifier l'interpréteur depuis une cellule**
   ```python
   import sys; print(sys.executable)
   ```
   Doit afficher : `/Users/vitt/code/world-model-from-scratch/.venv/bin/python`

5. **Validation complète en CLI** (sans VS Code)
   ```bash
   ./.venv/bin/python -m jupyter nbconvert --to notebook --execute test_plotly.ipynb \
     --output test_plotly_executed.ipynb \
     --ExecutePreprocessor.kernel_name=world-model-venv
   ```
   Si ça passe sans erreur, la stack Python/kernel est saine ; le souci vient alors du choix de kernel dans VS Code.

## Fichiers liés

- `.vscode/settings.json` — fixe l'interpréteur par défaut sur le venv
- `test_plotly.ipynb` — notebook de validation (à garder ou supprimer)
- `test_plotly_executed.ipynb` — sortie de la dernière exécution réussie
