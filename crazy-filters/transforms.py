"""
Dans ce module, tu trouveras des définitions de transformations d'images pixel à pixel.

Il est recommandé de lire le Readme ou d'avoir une explication sur les images numériques
avant de se lancer dans le code.
"""
import os
import numpy as np
import cv2

CANAL_ROUGE = 0
CANAL_VERT = 1
CANAL_BLEU = 2
CANAUX_RGB = [CANAL_ROUGE, CANAL_VERT, CANAL_BLEU]


def custom(image_array):
    """
    Cette fonction est associée au bouton Custom.
    1. Pour commencer, tu peux essayer d'autres fonctions définies plus bas (par exemple, `niveaux_de_gris`, `subimage`...) à la place
    de la fonction `rouge`.
    2. Essaie de définir et d'appeler ta propre fonction. Par exemple, définis une fonction
    `vert` en t'inspirant de la fonction `rouge`. Essaie de faire du `jaune`.
    3. Prend une fonction existante avec un filtre qui t'intéresse ('repeat', 'to_vintage', 'popart', 'cat_ears', 'colorize'...)
    Modifie le code dedans pour voir ce qui change ! Essaie de comprendre à quoi correspondent les différentes valeurs.
    4. Combine et/ou modifie différentes fonctions pour créer ton propre filtre !
    Par ex: peux-tu dessiner un drapeau ?

    :param image_array: image avant la transformation
    :return: image avec les pixels modifiés
    """
    return rouge(image_array)


"""
Partie 1: Jouer avec les canaux et les index.
Dans cette partie, tu vas apprendre assez pour pouvoir dessiner le drapeau français
sur la caméra. Tu peux essayer d'autres drapeaux tant que les dessins sont à base
de rectangles.
"""


def get_image_height(image_array):
    """
    Cette fonction donne la hauteur (height) d'une image
    :param image_array: image
    :return: la hauteur de image_array
    """
    return image_array.shape[0]


def get_image_width(image_array):
    """
    Cette fonction donne la largeur (width) d'une image
    :param image_array: image
    :return: le largeur de image_array
    """
    return image_array.shape[1]


def rouge(image_array):
    """
    Pour faire du rouge, on garde seulement les couleurs du canal rouge (qui
    correspond à l'index 0), et on met à 0 tous les pixels des canaux vert et bleu.
    :param image_array: image R, V, B
    :return: R, 0, 0
    """

    # h est la hauteur ('height') de l'image
    h = get_image_height(image_array)
    # w est la largeur ('width') de l'image
    w = get_image_width(image_array)
    image_array[0:h, 0:w, CANAL_VERT] = 0
    image_array[0:h, 0:w, CANAL_BLEU] = 0
    return image_array


def horizontal_subimage(image_array):
    """
    Cette fonction donne un exemple d'indexation sur l'axe vertical:
    on fixe un index en hauteur, et on sélectionne tous les pixels de la ligne.
    En prenant tous les pixels selon un index en hauteur, on prend toute
    la ligne.
    exemple: noircir les lignes de pixels de 0 à 9, pour les 3 canaux
    Questions:
     - est-ce que tu peux décaler la bande vers le bas ?
     - est-ce que tu peux faire une bande sur la moitié de l'image ?
     - est-ce que tu peux faire la bande avec la couleur que tu veux ?
    """
    w = get_image_width(image_array)
    image_array[0:10, 0:w, CANAUX_RGB] = 0
    return image_array


def vertical_subimage(image_array):
    """
    Cette fonction donne un exemple d'indexation sur l'axe horizontal:
    - les 10 colonnes de pixels qui sont sur le bord droit de l'image deviennent blanches
    - les colonnes numéro 10 à 19 deviennent blanches également
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    image_array[0:h, (w - 10):w, CANAUX_RGB] = 255
    image_array[0:h, 10:20, CANAUX_RGB] = 255
    return image_array


def subimage(image_array):
    """
    Dans cette fonction, on extrait un rectangle, donc on joue sur les
    index horizontaux et verticaux en même temps.
    Voilà une façon un peu compliquée de définir un rectangle rouge!
    Est-ce que tu peux deviner à peu près où va être le rectangle avant
    de lancer la fonction ?
    Essaie d'autres positions en restant DANS l'image!
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    image_array[(h - 150):(h - 100), 100:(w - 200), CANAUX_RGB] = 255, 0, 0
    return image_array


"""
Partie 2: jouer avec les couleurs
Ici on mélange les couleurs des différents canaux, et on transforme les pixels de différentes façons.
"""


def niveaux_de_gris(image_array):
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    r = image_array[0:h, 0:w, CANAL_ROUGE]
    v = image_array[0:h, 0:w, CANAL_VERT]
    b = image_array[0:h, 0:w, CANAL_BLEU]
    return r / 3. + v / 3. + b / 3.


def truncate_bounds(image):
    """
    Dans une image, les pixels doivent prendre des valeurs entre 0 et 255.
    Quand on fait des calculs, des valeurs peuvent sortir de cet intervalle.
    Cette fonction tronque les valeurs de pixels de façon
    à retourner une image valide.

    :param image: Image avec des valeurs entières quelconques
    :return: La même image, avec toutes les valeurs entre 0 et 255
    """
    return np.maximum(np.minimum(image, 255), 0)


def invert_image(image_array):
    """
    Transforme une image en négatif: le noir devient blanc et vice-versa.
    :param image_array: image normale
    :return: image en négatif
    """
    image_array = 255 - image_array
    return image_array


def to_sepia(image_array):
    """
    La couleur sepia est une transformation curieuse des pixels pour qu'ils soient tous
    un peu jaune/orangé.

    Les poids utilisés sont adaptés depuis: https://stackoverflow.com/questions/1061093/how-is-a-sepia-tone-created

    Essaie de les modifier, regarde comment ça change.
    est-ce que tu peux donner des teintes plus marron, plus vertes ? plus violettes ?
    de la couleur de ton choix ?

    :param image_array: image normale
    :return: image sepia
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)

    inputRed = image_array[0:h, 0:w, CANAL_ROUGE]
    inputGreen = image_array[0:h, 0:w, CANAL_VERT]
    inputBlue = image_array[0:h, 0:w, CANAL_BLEU]

    outputRed = ((inputRed * .393) + (inputGreen * .769) + (inputBlue * .189)) / 1.351
    outputGreen = ((inputRed * .349) + (inputGreen * .686) + (inputBlue * .168)) / 1.351
    outputBlue = ((inputRed * .272) + (inputGreen * .534) + (inputBlue * .131)) / 1.351

    image_array[0:h, 0:w, CANAL_ROUGE] = outputRed
    image_array[0:h, 0:w, CANAL_VERT] = outputGreen
    image_array[0:h, 0:w, CANAL_BLEU] = outputBlue
    truncate_bounds(image_array)
    return image_array


def colorize(image_array, color=(1, 1, 1, 1)):
    """
    Applique la couleur comme un filtre sur l'image. Attention, il faut passer un paramètre
    de couleur en plus de l'image !
    :param image_array: image d'entrée
    :param color: couleur du filtre à appliquer
    :return: image avec le filtre de la couleur
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    for canal in CANAUX_RGB:
        image_array[0:h, 0:w, canal] = image_array[0:h, 0:w, canal] * color[canal]
    return image_array


def colorfun(image_array, color=(1, 1, 1, 1)):
    """
    La couleur 'color' est celle choisie dans l'interface, qui renvoie des valeurs
    de pixels entre 0 et 1 (donc on multiplie par 255 pour avoir la bonne couleur).
    Cette fonction fait les opérations suivantes pour chaque canal R, V, B:
     - calculer la valeur 'mi' du pixel moyen
     - remplacer par 'color' tous les pixels qui ont une valeur plus grande que 'mi'
     En pratique, les couleurs les plus claires de l'image sont remplacées par 'color'.

    :param image_array: image d'entrée
    :param color: couleur à appliquer dans les zones claires
    :return: image colorisée
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    for c in range(3):
        mi = image_array[0:h, 0:w, c].mean()
        high = image_array[0:h, 0:w, c] >= mi
        image_array[0:h, 0:w, c][high] = color[c] * 255
    return image_array


"""
Partie 3: pour aller plus loin
Ces fonctions sont plus intéressantes à lire, essaie de modifier les paramètres
pour voir ce que ça change.
"""


def repeat(image_array, n_repeat=3):
    """
    Crée une image rapetissée en prenant 1 pixel sur 3 (`n_repeat`),
    et la répète 9 fois: 3 x en largeur, 3 x en hauteur.

    Essaie: n_repeat=2, ou 4, ...

    :param image_array: image d'origine
    :param n_repeat: nombre de répétitions sur chaque côté
    :return: image dupliquée
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    small_height = h // n_repeat
    small_width = w // n_repeat
    small_image = image_array[0:small_height * n_repeat:n_repeat, 0:small_width * n_repeat:n_repeat, CANAUX_RGB]
    new_image = np.zeros(image_array.shape, dtype=image_array.dtype)
    for i in range(n_repeat):
        for j in range(n_repeat):
            new_image[i * small_height:(i + 1) * small_height, j * small_width:(j + 1) * small_width, CANAUX_RGB] = small_image
    return new_image


def contour(image_array):
    """
    Extrait les 'contours' de l'image en couleur claire sur fond noir.

    Sur une zone uniforme, les pixels sont tous de la même couleur; si tu fais
    la différence entre un pixel et son voisin, c'est proche de 0 (donc noir).
    Un contour apparaît quand deux zones voisines ont des couleurs différentes;
    plus la différence est grande, plus le contour est marqué.
    Comme les contours sont à peu près les mêmes quel que soit le canal,
    les contours sont presque toujours blancs.

    Pour calculer le contour:
     - on décale l'image d'un pixel en hauteur et on fait la différence pour
    avoir les contours horizontaux (dy).
    - on répète l'opération en décalant l'image d'un pixel en largeur pour avoir dx,
    - et on fait un petit calcul pour mélanger les deux.

    Essaye d'observer directement:
        - dx: quelles sont ses valeurs maximum et minimum possible ?
        - np.abs(dx / 2)
        - (dx - np.min(dx)) / (np.max(dx) - np.min(dx)) * 255 -- est-ce qu'on voit bien le contour ?
    Intéressant non ? Mais on n'utilise que dx.
    Essaie maintenant de faire quelque chose de semblable en mélangeant dx et dy!
    Crée une nouvelle fonction 'embossage' (par exemple), c'est un effet assez différent.
    Tu peux créer des images intermédiaires pour t'aider.

    :return: les contours de l'image
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)

    imy1 = image_array[0:h - 1, 0:w, CANAUX_RGB] * 1.0
    imy2 = image_array[1:h, 0:w, CANAUX_RGB] * 1.0

    imx1 = image_array[0:h, 0:w - 1, CANAUX_RGB].astype(np.float)
    imx2 = image_array[0:h, 1:w, CANAUX_RGB] * 1.0

    # bordures en noir
    dx = np.zeros(image_array.shape)
    dy = np.zeros(image_array.shape)
    # dy
    dy[1:h, 0:w, CANAUX_RGB] = (imy2 - imy1)
    # dx
    dx[0:h, 1:w, CANAUX_RGB] = (imx1 - imx2)
    # une jolie formule pour les mélanger. dx^2 (au carré) s'écrirait dx ** 2 en Python,
    # mais c'est aussi simple d'écrire dx * dx.
    # np.sqrt = square root = racine carrée
    # (tiens ça ne te rappelerait pas Pythagore ?...)
    image_array[0:h, 0:w, CANAUX_RGB] = np.sqrt(dx * dx + dy * dy)

    return image_array


# Variable utilisée par la fonction 'temporal_difference' pour conserver l'image précédente de la vidéo
previous_image_array = None


def temporal_difference(image_array):
    """
    Calcule la différence entre 2 images successives de la vidéo (converties au préalable en niveaux de gris).

    Ça permet notamment de détecter des mouvements dans la vidéo.

    :param image_array: image d'origine
    :return: la différence (amplifiée) entre l'image courante et l'image précédente
    """

    # pour accéder et mettre à jour l'image précédente de la vidéo
    global previous_image_array
    # On convertit l'image en niveaux de gris, pour que le résultat pique moins les yeux.
    # Tu peux commenter cette ligne pour voir ce que ça donne sans.
    image_array = niveaux_de_gris(image_array)

    # La première fois que cette fonction est appelée, on n'a pas encore d'image précédente.
    if previous_image_array is None:
        previous_image_array = image_array

    # On applique l'opérateur "valeur absolue" sur la différence des deux images pour ne pas avoir de pixels "négatifs"
    difference = np.abs(image_array - previous_image_array)
    # L'image courante 'image_array' servira d'image précédente au prochain appel de cette fonction.
    previous_image_array = image_array
    # On amplifie la différence
    return truncate_bounds(10 * difference)


def posterisation(image_array):
    """
    Cette fonction utilise une fonction de la librairie `opencv`(`cv2.cvtColor`)
    pour changer l'espace de couleur. Tu peux trouver des infos détaillées sur
    Wikipedia. Ici on utilise l'espace HSV (au lieu de RGB).

    La postérisation consiste à réduire le nombre de couleurs utilisées pour représenter l'image.
    On pourrait juste arrondir les valeurs RGB à des valeurs multiples de 10 (essaie !)

    Mais c'est plus joli d'avoir beaucoup de teintes, et moins de variations
    clair/foncé.
    Ici on arrondit aux multiples de 20 sur la teinte (H), et aux multiples
    de 50 sur la saturation et la valeur (ou luminosité).

    :return: une image avec moins de couleurs.
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv_image[0:h, 0:w, CANAL_ROUGE] = (hsv_image[0:h, 0:w, CANAL_ROUGE] + 10.) // 20 * 20. + 10.
    hsv_image[0:h, 0:w, CANAL_VERT] =  (hsv_image[0:h, 0:w, CANAL_VERT] + 25.) // 50 * 50.
    hsv_image[0:h, 0:w, CANAL_BLEU] =  (hsv_image[0:h, 0:w, CANAL_BLEU] + 25.) // 50 * 50.
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    rgb_image = truncate_bounds(rgb_image)
    image_array = rgb_image
    return image_array


def popart_one(image_array, hue_delta=15):
    """
    :return: une image avec moins de couleurs. Comme la postérisation mais avec des paramètres différents.
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv_image[0:h, 0:w, CANAL_ROUGE] = (hsv_image[0:h, 0:w, CANAL_ROUGE] + 10.) // 30 * 30. + hue_delta
    hsv_image[0:h, 0:w, CANAL_VERT] =  (hsv_image[0:h, 0:w, CANAL_VERT] + 25.) // 100 * 100. + 50
    hsv_image[0:h, 0:w, CANAL_BLEU] =  (hsv_image[0:h, 0:w, CANAL_BLEU] + 25.) // 100 * 100. + 50
    hsv_image = truncate_bounds(hsv_image)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    image_array = rgb_image
    return image_array


def popart(image_array):
    """
    Applique la fonction popart 4 fois avec des paramètres un peu différents
    :return: une image avec moins de couleurs.
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    small_height = h // 2
    small_width = w // 2
    small_image = image_array[0:small_height * 2:2, 0:small_width * 2:2, CANAUX_RGB]
    new_image = np.zeros(image_array.shape, dtype=image_array.dtype)
    hue_delta = 0
    for i in range(2):
        for j in range(2):
            new_image[i * small_height:(i + 1) * small_height, j * small_width:(j + 1) * small_width, CANAUX_RGB] = popart_one(
                small_image,
                hue_delta)
            hue_delta += 45
    return new_image


def extract_contour(image_array, contours=None, use_alpha=True):
    """
    Cette fonction prend des listes de points (contours) et colorie
    de façon uniforme tous les pixels intérieurs d'un canal.
    :param image_array: image d'entrée.
    :param contours: liste des points de contour. Par ex: [[(0,0), (6,30), (50,100)]]
    le contour est fermé en revenant au premier point.
    :param use_alpha: si True, colorie le canal alpha, sinon rouge.
    :return: une image avec la forme définie par contour d'une couleur différente.
    """
    if contours is None or len(contours) < 1:
        return image_array
    if use_alpha:
        canal = 3
    else:
        canal = 1
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    new_image = image_array[0:h, 0:w, canal].copy()
    cv2.drawContours(new_image, np.array(contours).astype(int), -1, 100, -1)
    image_array[0:h, 0:w, canal] = new_image
    return image_array


def test_contours(image_array):
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    # dessine une étoile jaune d'or
    star_contour = create_star_shape(75, 75, 19, 50)
    color = (255, 197, 25)
    im1 = draw_color_mask(image_array, star_contour, color)
    w2, h2 = w, h

    # dessine des couleurs en arc-en-ciel
    thickness = 50
    colors = [(255, 0, 0), (255, 180, 50), (255, 255, 0)]
    for col in colors:
        contour2 = create_polygon_shape(w, h, w2, 100, scale_y=h2 / w2)
        im1 = draw_color_mask(im1, contour2, col, thickness=thickness)
        w2 -= thickness + 1
        h2 -= thickness + 1
    return im1


def draw_contours(image_array, contour_points, color, thickness=-1):
    """
    :param image_array: image sur laquelle dessiner la forme
    :param contour_points: liste des points formant le contour [(x, y), ...]
    :param color: couleur à utiliser pour dessiner la forme (R, G, B) dans [0, 255]
    :param thickness: -1 pour remplir la forme, sinon épaisseur de la ligne
    :return: une nouvelle image avec la forme dessinée par-dessus l'image originale
    """
    new_im = image_array.copy()
    cv2.drawContours(new_im, [np.array(contour_points).astype(np.int)], 0, color, thickness)
    return new_im


def draw_color_mask(image_array, contours, color, thickness=-1):
    """
    Un peu comme draw_contours, sauf que cette fois-ci la forme dessinée avec la couleur est utilisée comme
    un masque (comme colorize)
    """
    mask = np.ones_like(image_array, dtype=np.uint8) * 255
    mask = draw_contours(mask, contours, color, thickness) / 255
    return image_array * mask


def create_star_shape(pos_x, pos_y, min_radius, max_radius, nb_branch=5, rotate=0):
    """
    Retourne une liste de points pour dessiner une étoile centrée en (pos_x, pos_y),
    avec des pointes crée entre min_radius et max_radius, et nb_branch pointes.
    Le paramètre rotate (en radians) permet de tourner l'étoile.
    """
    angle = -np.pi / 2 + rotate
    px = pos_x + max_radius * np.cos(angle)
    py = pos_y + max_radius * np.sin(angle)
    list_of_points = [(px, py)]
    for _ in range(nb_branch):
        angle += np.pi / nb_branch
        px = pos_x + min_radius * np.cos(angle)
        py = pos_y + min_radius * np.sin(angle)
        list_of_points.append((px, py))
        angle += np.pi / nb_branch
        px = pos_x + max_radius * np.cos(angle)
        py = pos_y + max_radius * np.sin(angle)
        list_of_points.append((px, py))
    return list_of_points


def create_polygon_shape(pos_x, pos_y, radius, nb_sides=5, rotate=0, scale_y=1.):
    """
    Retourne une liste de points pour dessiner un polygone dont le centre est en position (pos_x, pos_y),
    avec nb_sides côtés, et un "rayon" radius (distance du centre à chaque sommet).
    Pour faire un cercle, il suffit d'augmenter le nombre de côtés (par ex. 100).
    Le paramètre scale_y permet de réduire ou d'augmenter la hauteur du polygone de ce facteur.
    Le paramètre rotate applique une rotation à la forme.
    """
    angle = -np.pi / 2 + rotate
    px = pos_x + radius * np.cos(angle)
    py = pos_y + radius * np.sin(angle) * scale_y
    list_of_points = [(px, py)]
    for _ in range(nb_sides):
        angle += 2 * np.pi / nb_sides
        px = pos_x + radius * np.cos(angle)
        py = pos_y + radius * np.sin(angle) * scale_y
        list_of_points.append((px, py))
    return list_of_points


def to_vintage(image_array):
    """
    Combine plusieurs effets (ici des fonctions) pour obtenir un effet "vintage".
    """
    image_array = add_noise(image_array)
    image_array = to_sepia(image_array)
    return add_vignetting(image_array, 75, 75)


def add_noise(image_array):
    """
    Rajoute du bruit sur une image.

    Le bruit se traduit en partie par la présence de 'grain' sur l'image.

    Ici le grain est représenté de facon simpliste par l'addition ou la soustraction d'une petite valeur sur chaque pixel.
    Chaque valeur est choisie aleatoirement avec une forte chance d'etre proche de zero (suivant une loi normale).
    Pour la loi normale, voir par exemple: https://fr.wikipedia.org/wiki/Loi_normale
    """
    h = get_image_height(image_array)
    w = get_image_width(image_array)
    noise = np.random.normal(loc=0, scale=8, size=(h, w))

    inputRed = image_array[0:h, 0:w, CANAL_ROUGE]
    inputGreen = image_array[0:h, 0:w, CANAL_VERT]
    inputBlue = image_array[0:h, 0:w, CANAL_BLEU]

    noisy_image = np.ndarray(image_array.shape, dtype=float)

    noisy_image[0:h, 0:w, CANAL_ROUGE] = inputRed + noise
    noisy_image[0:h, 0:w, CANAL_VERT] = inputGreen + noise
    noisy_image[0:h, 0:w, CANAL_BLEU] = inputBlue + noise

    return truncate_bounds(noisy_image)


def add_vignetting(image_array, spread_factor_h, spread_factor_w):
    """
    Le vignettage est un effect d' assombrissement des bords d'une image qu'on observe souvent sur des photos anciennes
    mais qui existe aussi sur des appareils récents.

    PLus d' info:   https://fr.wikipedia.org/wiki/Vignettage

    :param image_array: l'image sur laquelle il faut rajouter du vignettage.
    :param spread_factor_h: la taille suivant la hauteur de la partie qui reste "claire"
    :param spread_factor_w: la taille suivant la largeur de la partie qui reste "claire"
    :return: l'image avec du vignettage.
    """

    h = get_image_height(image_array)
    w = get_image_width(image_array)

    gaussian = draw_gaussian(h, w, spread_factor_h * h, spread_factor_w * w)

    inputRed = image_array[0:h, 0:w, CANAL_ROUGE]
    inputGreen = image_array[0:h, 0:w, CANAL_VERT]
    inputBlue = image_array[0:h, 0:w, CANAL_BLEU]

    outputRed = inputRed * gaussian
    outputGreen = inputGreen * gaussian
    outputBlue = inputBlue * gaussian

    image_array[0:h, 0:w, CANAL_ROUGE] = outputRed
    image_array[0:h, 0:w, CANAL_VERT] = outputGreen
    image_array[0:h, 0:w, CANAL_BLEU] = outputBlue
    truncate_bounds(image_array)
    return image_array


def draw_gaussian(height, width, spread_h, spread_w):
    """
    Utilisé pour dessiner le vignettage.

    La formule est celle d'une gaussienne, voir:
    https://fr.wikipedia.org/wiki/Fonction_gaussienne
    """
    size_h = height / 2
    size_w = width / 2
    coordinate_h, coordinate_w = np.mgrid[-size_h:size_h, -size_w:size_w]
    g = np.exp(-(coordinate_h ** 2 / float(spread_h) + coordinate_w ** 2 / float(spread_w)))
    normalized_g = g / g.max()
    return normalized_g


def peephole_effect(image_array):
    """
    Toc toc toc, qui est la ?

    On simule ici le fait de regarder par un petit trou (de la porte par exemple).
    """
    return add_vignetting(image_array, 10, 10)


"""
Utiliser la détection de visage d'OpenCV pour faire un filtre de type snapchat.
Les fonctions ci-dessous utilisent un quatrième canal, "alpha",  en plus de R, G, B, pour gérer la transparence.
"""


def test_paste_image(image_array):
    myim = load_image('images/kitten3.jpg')
    paste_image(image_array, myim, 10, 10, fit_width=200, make_transparent=(255, 255, 250))
    del myim
    return image_array


def load_image(image_path, with_alpha=True, read_as_bgr=True):
    """
    Il y a une fonction OpenCV pour lire des images sur le disque
    OpenCV renvoie les canaux dans l'ordre B, G, R donc on les remet
    dans l'ordre R, G, B.
    Tu peux aussi décider si tu veux ou non lire avec un canal alpha.
    :param image_path:
    :param with_alpha:
    :param read_as_bgr:
    :return:
    """
    flag = -1 if with_alpha else 0
    if os.path.dirname(__file__) != os.getcwd():
        image_path = os.path.join(os.path.dirname(__file__), image_path)
    myim = cv2.imread(image_path, flag)
    if read_as_bgr:
        if myim.shape[2] == 4:
            myim = myim[:, :, [CANAL_BLEU, CANAL_VERT, CANAL_ROUGE, 3]]
        else:
            myim = myim[:, :, [CANAL_BLEU, CANAL_VERT, CANAL_ROUGE]]
    return myim


def paste_image(image_array,
                image_to_paste,
                pos_x=0,
                pos_y=0,
                fit_width=None,
                fit_height=None,
                make_transparent=None):
    """
    Copie image_to_paste dans image_array, à la position (pos_x, pos_y), et en la
    redimensionnant à une largeur fit_width (ou une largeur fit_height si fit_width=None)

    Utilise le canal alpha, ou la couleur `make_transparent` pour coller avec un fond transparent.
    """
    image_height = image_array.shape[0]
    image_width = image_array.shape[1]

    if fit_height is None and fit_width is None:
        # essaie de déterminer automatiquement comment redimensionner l'image, en hauteur ou en largeur
        arr_ratio = image_height / image_width
        paste_ratio = image_to_paste.shape[0] / image_to_paste.shape[1]
        if arr_ratio < paste_ratio:  # fit en hauteur
            fit_height = image_height - pos_y
        else:
            fit_width = image_width - pos_x

    if fit_height is None:
        new_height = int(fit_width * image_to_paste.shape[0] / image_to_paste.shape[1])
        new_width = fit_width
    else:
        new_height = fit_height
        new_width = int(fit_height * image_to_paste.shape[1] / image_to_paste.shape[0])

    old_image_patch = image_array[pos_y:pos_y + new_height, pos_x:pos_x + new_width, :]
    resized_image = cv2.resize(image_to_paste, (new_width, new_height))
    if resized_image.shape[2] == 4 and np.allclose(resized_image[:, :, 3].mean(), 255):  # il y a un canal alpha
        resized_image = resized_image[:, :, :3]
    if resized_image.shape[2] == 4 and resized_image[:, :, 3].mean() < 255:  # il y a un canal alpha
        img_mask = resized_image[:, :, 3] / 255
        img_mask = img_mask[:, :, np.newaxis]
        patch_to_paste = img_mask * resized_image[:, :, :3] + (1 - img_mask) * old_image_patch
    elif make_transparent is not None:
        img_mask = np.all(np.isclose(resized_image, make_transparent, atol=5), axis=2)
        img_mask = img_mask[:, :, np.newaxis].astype(float)
        patch_to_paste = (1 - img_mask) * resized_image + img_mask * old_image_patch
    else:
        patch_to_paste = resized_image

    # retailler patch_to_paste si ça déborde de l'image image_array
    if pos_y < 0:
        patch_to_paste = np.delete(patch_to_paste, range(-pos_y), axis=0)
        pos_y = 0
        new_height = patch_to_paste.shape[0]
    if pos_x < 0:
        patch_to_paste = np.delete(patch_to_paste, range(-pos_x), axis=1)
        pos_x = 0
        new_width = patch_to_paste.shape[1]

    if pos_y + new_height > image_height:
        patch_to_paste = np.delete(patch_to_paste, range(image_height - pos_y, new_height), axis=0)
    if pos_x + new_width > image_width:
        patch_to_paste = np.delete(patch_to_paste, range(image_width - pos_x, new_width), axis=1)

    new_height = patch_to_paste.shape[0]
    new_width = patch_to_paste.shape[1]

    image_array[pos_y:pos_y + new_height, pos_x:pos_x + new_width, :] = patch_to_paste
    return image_array


def glasses(img):
    """
    Un exemple de fonction pour afficher des lunettes. Pour utiliser une
    autre image:
    - trouve une image et sauvegarde-la dans le dossier 'mask'
    - change le chemin de l'image utilisée ci-dessous
    - change les paramètres `relative_position` et `relative_width` pour positionner l'image
    correctement par rapport au rectangle entourant le visage (relative position peut
    être négatif)

    :param img: image en entrée (par ex. une frame de la vidéo)
    :return: image avec des lunettes ajoutées par-dessus
    """
    glass_image = load_image('masks/black-glasses.png', with_alpha=True)
    return face_overlay(img, glass_image, relative_position=0.21)


def face_overlay(img, overlay=None, relative_position=0.2, relative_width=1.):
    """
    Cette fonction utilise la fonction de détection de visages d'OpenCV
    et colle une image sur les visages détectés

    :param image_array: image avant la transformation
    :return: image avec les pixels modifiés
    """
    datadir = os.path.dirname(cv2.__file__)
    face_cascade = cv2.CascadeClassifier(datadir + '/data/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.int32)  # required to draw rectangle
    # applique la fonction qui détecte tous les visages dans l'image
    # chaque "face" est un rectangle défini par la position du coin haut-gauche (x, y) et sa taille
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        # pour chaque visage, colle l'image un peu au-dessus du rectangle
        # tu peux afficher le rectangle en décommentant la ligne suivante
        if overlay is None:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            continue

        new_y = max(0, int(y + relative_position * h))
        new_width = int(w * relative_width)
        new_x = x - (new_width - w) // 2
        paste_image(img, overlay, new_x, new_y, fit_width=new_width, make_transparent=(255, 255, 255))

    return img


def big_eyes(img, scale=1.6):
    """
    Cette fonction utilise la fonction de détection de yeux ouverts d'OpenCV pour les agrandir
    :param image_array: image avant la transformation
    :return: image avec les pixels modifiés
    """
    datadir = os.path.dirname(cv2.__file__)
    face_cascade = cv2.CascadeClassifier(datadir + '/data/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # applique la fonction qui détecte tous les yeux ouverts dans l'image
    # chaque "eye" est un rectangle défini par la position du coin haut-gauche (x, y) et sa taille
    eyes = face_cascade.detectMultiScale(gray, 1.3, 5)

    # On récupère la sous-image de chaque oeil avant de modifier l'image avec les yeux agrandis
    eye_images = []
    for x, y, w, h in eyes:
        eye_images.append((img[y:y + h, x:x + w, CANAUX_RGB], x, y, w, h))

    # On colle dans l'image un agrandissement des yeux
    for eye, x, y, w, h in eye_images:
        new_height = int(h * scale)
        new_y = y - (new_height - h) // 2
        new_width = int(w * scale)
        new_x = x - (new_width - w) // 2
        paste_image(img, eye, new_x, new_y, fit_width=new_width)

    return img


def carrousel_transfo(image_array, ticking):
    """
    Cette fonction permet de boucler sur une liste prédéfinie de fonctions.

    :param image_array:
    :param ticking: une liste de taille 1, dont l'élément compte le nombre de frames affichées depuis
        le dernier clic sur le bouton.
    :return: une image transformée
    """

    def draw_rect_horiz(im):
        w = get_image_width(image_array)
        im[100:150, 0:w, CANAUX_RGB] = 0
        return im

    def draw_rect_horiz2(im):
        w = get_image_width(image_array)
        im[120:170, 0:w, CANAUX_RGB] = 0
        return im

    def draw_rect_vert(im):
        h = get_image_height(image_array)
        im[0:h, 100:150, CANAUX_RGB] = 255
        return im

    def draw_rect_vert2(im):
        h = get_image_height(image_array)
        im[0:h, 120:170, CANAL_VERT] = 255
        return im

    available_transfos = (
        rouge,
        draw_rect_horiz,
        draw_rect_horiz2,
        draw_rect_vert,
        draw_rect_vert2,
        subimage,
        niveaux_de_gris,
        invert_image,
        to_sepia,
        repeat,
        test_contours,
        peephole_effect,
        to_vintage,
        contour,
        popart_one,
        popart,
        temporal_difference,
        temporal_difference,
        face_overlay,
        face_overlay,  # keep longer hack
        glasses,
        glasses,
        glasses,
        big_eyes,
        big_eyes,
        big_eyes
    )

    function_index = (ticking[0] // 10) % len(available_transfos)
    return available_transfos[function_index](image_array)
