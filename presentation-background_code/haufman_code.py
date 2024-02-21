import numpy as np
import heapq
from collections import defaultdict
from scipy.fft import dct, idct

def calculate_bin(value):
  return len(value)//8

def calculate_numpy_image_size(image):
    # Obtenir les dimensions de l'image
    height, width, channels = image.shape
    # Chaque pixel utilise 'channels' octets (typiquement 3 pour RGB)
    size_in_bytes = height * width * channels
    return size_in_bytes


def calculate_encrypted_data_size(encrypted_data):
    size_in_bytes = 0
    # La taille en bits est égale à la longueur de la chaîne
    size_in_bits = calculate_bin(encrypted_data)
    return size_in_bytes

def calculate_huffman_dict_size(huffman_dicts):
    key_size = 0
    value_size = 0
    for huffman_dict in huffman_dicts:
      for h in huffman_dict:
        for key, value in h.items():
            # Taille de la clé (entier) + taille de la valeur (chaîne binaire)
            key_size += 4 #4 octets par int et un int par key
            value_size += calculate_bin(value)  # value en binaire
    return key_size + value_size

def calculate_huffman(encoded_data, huffman_dicts):
  return calculate_encrypted_data_size(encoded_data)+calculate_huffman_dict_size(huffman_dicts)

def sub_cr(Cb):
    Cb_down = np.zeros((int(Cb.shape[0]/2), int(Cb.shape[1]/2)))
    for i in range(0, Cb.shape[0], 2):
        for j in range(0, Cb.shape[1], 2):
            block = Cb[i:i+2, j:j+2]
            Cb_down[int(i/2), int(j/2)] = np.mean(block)
    return Cb_down

def split_into_blocks_numpy(image_channel, block_size=8):

    height, width = image_channel.shape

    # Calculez le nombre de blocs en hauteur et en largeur
    num_blocks_height = -(-height // block_size)  # Round up division
    num_blocks_width = -(-width // block_size)  # Round up division

    # Paddage de l'image si nécessaire
    padded_height = num_blocks_height * block_size
    padded_width = num_blocks_width * block_size

    padded_image = np.pad(image_channel, ((0, padded_height - height), (0, padded_width - width)), mode='edge')
    #le mode="edge" permet de répéter les bords du bloc en cas de division non-entière

    # Reshape pour obtenir les blocs
    blocks = padded_image.reshape(num_blocks_height, block_size, num_blocks_width, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

    return blocks

def dct2D(x):
    tmp = dct(x, type=2 ,norm='ortho').transpose()#on fait une dct sur les colonnes puis on transpose
    return dct(tmp, type=2 ,norm='ortho').transpose()#on fait une dct sur les lignes puis on retranspose

def quantize(block, quantization_matrix):
    quantized_block = np.round(np.divide(block, quantization_matrix)).astype(np.int64)
    return quantized_block

def dequantize(quantized_block, quantization_matrix):
    block = np.multiply(np.array(quantized_block).astype(float), quantization_matrix.astype(float))
    return block


# Tables de quantification classiques utilisées en JPEG
luma_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 40, 51, 61],
    [14, 13, 15, 20, 25, 38, 51, 60],
    [14, 17, 19, 22, 29, 44, 55, 64],
    [18, 21, 26, 34, 37, 52, 61, 66],
    [24, 26, 32, 40, 48, 58, 66, 70],
    [29, 34, 39, 48, 56, 64, 72, 76],
    [42, 49, 54, 64, 70, 77, 81, 83]
])

chroma_table = np.array([
    [17, 18, 24, 40, 51, 61, 67, 72],
    [18, 21, 26, 43, 55, 67, 72, 76],
    [21, 25, 30, 47, 62, 72, 77, 80],
    [24, 29, 34, 52, 65, 74, 79, 82],
    [27, 32, 37, 56, 70, 78, 81, 85],
    [30, 35, 39, 59, 73, 81, 84, 87],
    [53, 62, 72, 84, 92, 99, 105, 110],
    [60, 69, 77, 90, 98, 105, 110, 113]
])


# Les seuils sont basés sur des recommandations courantes pour le réglage de la qualité lors de la compression JPEG (Internet).

def quality_factor_matrix(quality, quantization_matrix):

# Facteur de qualité entre 1 et 100
  if quality < 50:
      scale = 5000 / quality
  else:
      scale = 200 - 2 * quality

# On ajuste les tables de quantification en fonction du facteur de qualité
  for i in range(8):
      for j in range(8):
          quantization_matrix[i][j] = max(1, min(255, int((quantization_matrix[i][j] * scale + 50) / 100)))

  return quantization_matrix


def quantize_image(image, quantization_matrix):
  quantized = np.copy(image)
  lenght = len(image)
  for i in range(lenght):
    quantized[i] = quantize(image[i], quantization_matrix)
  return quantized

def dequantize_image(q_image, quantization_matrix):
  quantized = np.copy(q_image)
  lenght = len(q_image)
  for i in range(lenght):
    quantized[i] = dequantize(q_image[i], quantization_matrix)
  return quantized

def zigzag_order(matrix):
    # Obtient le nombre de lignes et de colonnes de la matrice
    rows, cols = matrix.shape

    # Initialisez une liste pour stocker le résultat du parcours en zigzag
    result = []

    # Parcourt de la diagonale supérieure gauche à la diagonale inférieure droite
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            # Si l'indice est pair, parcourt de haut en bas sur la diagonale
            for j in range(max(0, i - cols + 1), min(rows, i + 1)):
                result.append(matrix[j][i - j])
        else:
            # Si l'indice est impair, parcourt de bas en haut sur la diagonale
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                result.append(matrix[j][i - j])

    # Retourne la liste résultante
    return result

# Fonction pour encoder les valeurs RLE
def rle_encode(data):
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        # Utiliser np.all pour comparer tous les éléments
        if np.all(data[i] == data[i - 1]):
            count += 1
        else:
            encoded_data.append((data[i - 1], count))
            count = 1
    # Ajouter la dernière séquence
    encoded_data.append((data[-1], count))
    return encoded_data

# Fonction pour encoder les valeurs RLE avec Huffman
def huffman_encode(rle_data):
    # Création d'un dictionnaire pour compter la fréquence de chaque valeur
    frequency_dict = defaultdict(int)

    for value, count in rle_data:
        # Convertir les valeurs numpy.ndarray en tuples pour les rendre hachables
        if isinstance(value, np.ndarray):
            value = tuple(value)
        frequency_dict[value] += count

    # Création d'un tas (heap) à partir du dictionnaire de fréquence
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency_dict.items()]
    heapq.heapify(heap)

    # Si le tas n'a qu'un seul élément, lui assigner un code binaire fixe
    if len(heap) == 1:
        heap = [[heap[0][0], [heap[0][1][0], '0']]]

    # Combinaison des deux éléments les moins fréquents jusqu'à ce que le tas ne contienne qu'un seul élément
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Création d'un dictionnaire Huffman à partir de l'arbre
    huffman_dict = {symbol: code for symbol, code in heap[0][1:]}

    # Encodage des données RLE en utilisant le dictionnaire Huffman
    encoded_data = "".join(huffman_dict[tuple(value)] if isinstance(value, np.ndarray) else huffman_dict[value] for value, count in rle_data for _ in range(count))

    return encoded_data, huffman_dict

def encode_blocks(blocks):
    zigzag_data, rle_data, encoded_data, huffman_dicts = [], [], [], []
    for block in blocks:
        # Zigzag Order
        zz_data = zigzag_order(block)
        zigzag_data.append(zz_data)

        # RLE Encode
        rle_data_block = rle_encode(zz_data)
        rle_data.append(rle_data_block)

        # Huffman Encode
        encoded_block, huffman_dict = huffman_encode(rle_data_block)
        encoded_data.append(encoded_block)
        huffman_dicts.append(huffman_dict)

    return encoded_data, huffman_dicts

#decodage
def huffman_decode(encoded_data, huffman_dict):
    # Inverser le dictionnaire Huffman
    decode_dict = {code: symbol for symbol, code in huffman_dict.items()}

    decoded_values = []
    temp_code = ""
    for bit in encoded_data:
        temp_code += bit
        if temp_code in decode_dict:
            symbol = decode_dict[temp_code]
            decoded_values.append(symbol)
            temp_code = ""

    return decoded_values

def rle_decode(encoded_data):
    decoded_data = np.array([])
    for value, count in encoded_data:
        decoded_data = np.append(decoded_data, np.repeat(value, count))
    return decoded_data

def zigzag_decode(zigzag_data, rows, cols):
    matrix = [[0] * cols for _ in range(rows)]
    index = 0
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            for j in range(max(0, i - cols + 1), min(rows, i + 1)):
                matrix[j][i - j] = zigzag_data[index]
                index += 1
        else:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                matrix[j][i - j] = zigzag_data[index]
                index += 1
    return matrix

def decode_blocks(encoded_blocks, huffman_dicts, block_size = 8):
    decoded_blocks = []
    for block_index, encoded_block in enumerate(encoded_blocks):
        huffman_dict = huffman_dicts[block_index]

        # Huffman Decode
        rle_data = huffman_decode(encoded_block, huffman_dict)

        # Zigzag Reorder to Block
        block = zigzag_decode(rle_data, block_size, block_size)
        decoded_blocks.append(block)

    return decoded_blocks

def idct2D(x):
    tmp = idct(x, type=2, norm='ortho').transpose()
    return idct(tmp, type=2, norm='ortho').transpose()

def block_into_image(channel_block, original_shape):

    nb_blocks, length, width = channel_block.shape
    channel=channel_block.reshape((nb_blocks,length*width))
    channel=channel.reshape((original_shape[0],original_shape[1]))
    return(channel)


def combine_blocks(blocks, original_shape):
    """
    Combine les blocs dans une seule image en inversant le processus de découpage.

    Parameters:
    - blocks: Les blocs résultants du processus de découpage.
    - original_shape: La forme originale de l'image avant le découpage.

    Returns:
    - L'image reconstituée.
    """

    block_size = blocks.shape[1]  # Taille du bloc
    num_blocks_height = original_shape[0] // block_size
    num_blocks_width = original_shape[1] // block_size

    # Reshape pour obtenir les blocs dans le format original
    reshaped_blocks = blocks.reshape(num_blocks_height, num_blocks_width, block_size, block_size)

    # Transposer les axes pour obtenir la forme correcte
    transposed_blocks = reshaped_blocks.transpose(0, 2, 1, 3)

    # Reshape pour obtenir l'image originale
    reconstructed_image = transposed_blocks.reshape(original_shape)

    return reconstructed_image

def unsub_cb(Cb_down):
  val_int = 0
  Cb=np.zeros((int(Cb_down.shape[0]*2),int(Cb_down.shape[1]*2)))
  for i in range(int(Cb_down.shape[0])):
    colonne_max = (i+1)*2-1
    for j in range(int(Cb_down.shape[1])):
      ligne_max = (j+1)*2-1
      val_int = Cb_down[i,j]
      Cb[colonne_max, ligne_max] = val_int
      Cb[colonne_max, ligne_max-1] = val_int
      Cb[colonne_max-1, ligne_max] = val_int
      Cb[colonne_max-1, ligne_max-1] = val_int
  return Cb

def calculate_RGB(Y, Cb, Cr):
  R_result = 0.990 * Y  + 0.008 * (Cb - 128)+1.409*(Cr - 128)
  V_result =  0.990 * Y  -0.331 * (Cb - 128)-0.707*(Cr - 128)
  B_result =  0.990 * Y  +1.784 * (Cb - 128)+0.013*(Cr - 128)
  return(R_result,V_result,B_result)

def encode_image_to_huffman(image):
    list_result = []
    #image en output
    V = image[:,:,0]
    B = image[:,:,1]
    R = image[:,:,2]
    #using coefficients of https://www.youtube.com/watch?v=Kv1Hiv3ox8I
    Y=(0.3*R+0.6*V+0.11*B)
    Cb =np.array(-0.17*R +0.5 *B - 0.33*V+128)
    Cr =np.array( 0.5*R - 0.42*V- 0.08*B+128)
    #sous echantillonnage de l'image
    Cb_down = sub_cr(Cb)
    Cr_down = sub_cr(Cr)
    list_result.append(calculate_numpy_image_size(image))

    Y_blocks = split_into_blocks_numpy(Y)
    Cb_down_blocks=split_into_blocks_numpy(Cb_down)
    Cr_down_blocks=split_into_blocks_numpy(Cr_down)

    Y_DCT=dct2D(Y_blocks)
    Cb_down_DCT=dct2D(Cb_down_blocks)
    Cr_down_DCT=dct2D(Cr_down_blocks)

    Y_quantized = quantize_image(Y_DCT, luma_table)
    Cb_quantized = quantize_image(Cb_down_DCT, chroma_table)
    Cr_quantized = quantize_image(Cr_down_DCT, chroma_table)

    encoded_data = []
    huffman_dicts = []

    # Encoder les données Y
    encoded_Y, huffman_Y = encode_blocks(Y_quantized)
    encoded_data.append(encoded_Y)
    huffman_dicts.append(huffman_Y)

    # Encoder les données Cb
    encoded_Cb, huffman_Cb = encode_blocks(Cb_quantized)
    encoded_data.append(encoded_Cb)
    huffman_dicts.append(huffman_Cb)

    # Encoder les données Cr
    encoded_Cr, huffman_Cr = encode_blocks(Cr_quantized)
    encoded_data.append(encoded_Cr)
    huffman_dicts.append(huffman_Cr)

    list_result.append(calculate_huffman(encoded_data, huffman_dicts))

    return encoded_data, huffman_dicts, list_result

def decode_huffman_to_image(encoded_data, huffman_dicts, original_shape):
    print(original_shape)
    # Extraire les données encodées et les dictionnaires Huffman pour chaque composante
    encoded_Y, encoded_Cb, encoded_Cr = encoded_data
    huffman_Y, huffman_Cb, huffman_Cr = huffman_dicts

    # Décoder les données Y
    decoded_Y = decode_blocks(encoded_Y, huffman_Y)

    # Décoder les données Cb
    decoded_Cb = decode_blocks(encoded_Cb, huffman_Cb)

    # Décoder les données Cr
    decoded_Cr = decode_blocks(encoded_Cr, huffman_Cr)

    # La fonction a déja été définie dans la partie quantification
    dequantized_Y = dequantize_image(decoded_Y, luma_table)
    dequantized_Cb = dequantize_image(decoded_Cb, chroma_table)
    dequantized_Cr = dequantize_image(decoded_Cr, chroma_table)

    Y_IDCT_blocks=idct2D(dequantized_Y)
    Cb_down_IDCT_blocks=idct2D(dequantized_Cb)
    Cr_down_IDCT_blocks=idct2D(dequantized_Cr)

    #Y_IDCT = block_into_image(Y_IDCT_blocks, original_shape=(original_shape[0], original_shape[1]))
    #Cb_down_IDCT = block_into_image(Cb_down_IDCT_blocks, original_shape=(int(original_shape[0]/2), int(original_shape[1]/2)))
    #Cr_down_IDCT = block_into_image(Cr_down_IDCT_blocks, original_shape=(int(original_shape[0]/2), int(original_shape[1]/2)))

    Y_IDCT = combine_blocks(Y_IDCT_blocks, original_shape=(original_shape[0], original_shape[1]))
    Cb_down_IDCT = combine_blocks(Cb_down_IDCT_blocks, original_shape=(int(original_shape[0]/2), int(original_shape[1]/2)))
    Cr_down_IDCT = combine_blocks(Cr_down_IDCT_blocks, original_shape=(int(original_shape[0]/2), int(original_shape[1]/2)))

    un_cb = unsub_cb(Cb_down_IDCT)
    un_cr = unsub_cb(Cr_down_IDCT)

    # Calcul des composantes RVB
    R_result, V_result, B_result = calculate_RGB(Y_IDCT, un_cb, un_cr)

    image_result=np.stack([V_result, B_result,R_result],axis=-1)

    return image_result