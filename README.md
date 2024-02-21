# Compression JPEG
## Présentation
Ce projet a été réalisé dans le cadre du cours d'_Image_, afin de comprendre le fonctionnement de la compression JPEG.

Les membres du groupe par ordre alphabétique de leur nom de famille sont :
- Alexandre Blain
- Paul Corbalan
- Johan Lagardère
- Younès Loulidi

Les principaux fichiers sont :
- Notebook Jupyter ayant pour but de servir de rapport interactif : `notebook.ipynb`
- Support de présentation interactif : `presentation.ipynb`

## Installation
### Environment Python
#### Utilisation de Python 3.8
1. Création de l'environnement virtuel avec _conda_.
    ```shell
    conda create -n jpeg-compression python=3.8
    ```
2. Activation de l'environnement virtuel.
    ```shell
    conda activate jpeg-compression
    ```

#### Installation des librairies
```shell
pip install -r requirements.txt
```

### Extension Google Chrome
1. Ouvrir _Google Chrome_, _Chromium_ ou _Microsoft Edge_.
2. Ouvrir les extensions.
3. Activer le mode développeur.
4. Charger l'extension non empaquetée à partir du dossier `extension_chrome`.


## Utilisation
### Extension Google Chrome
#### Lancement de l'application Flask
1. Changer de répertoire vers `presentation-background_code`.
    ```shell
    cd presentation-background_code
    ```
2. Lancer l'application Flask.
    ```shell
    python flask_image.py
    ```

#### Obtenir l'encodage d'Huffman d'une image en fichier texte
1. Ouvrir _Google Chrome_, _Chromium_ ou _Microsoft Edge_.
2. Ouvrir l'extension.
3. Sélectionner une image.
4. Appuyer sur le bouton _Encoder_.
5. Télécharger le fichier texte.
    - La taille sera affichée en dessous du bouton _Encoder_.

#### Obtenir l'image à partir d'un encodage d'Huffman
1. Ouvrir _Google Chrome_, _Chromium_ ou _Microsoft Edge_.
2. Ouvrir l'extension.
3. Sélectionner le fichier texte contenant l'encodage d'Huffman.
4. Appuyer sur le bouton _Decoder_.
5. Télécharger l'image.
    - La taille sera affichée en dessous du bouton _Decoder_.
