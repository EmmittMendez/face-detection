import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Pedimos la ruta de la carpeta donde se encuentran las imagenes
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--output", required=False, help="path to the output image file")
ap.add_argument("-k", "--k", required=True, help="k value")     # Valor para la ecualización
ap.add_argument("-g", "--gamma", required=True, help="gamma value")  # Valor para la corrección gamma
ap.add_argument("-b", "--low", required=True, help="a_low value")   # Valor porcentual de a_low
ap.add_argument("-a", "--high", required=True, help="a_high value")  # Valor porcentual de a_high
ap.add_argument("-m", "--min", required=True, help="a_min value")   # Valor a min
ap.add_argument("-n", "--max", required=True, help="a_max value")   # Valor a max
ap.add_argument("-t", "--typeb", required=True, help="Type of blurring")    # Tipo de suavizado
ap.add_argument("-l", "--ksize", required=True, help="Size of the kernel")  # Tamaño del kernel
args = vars(ap.parse_args())

imageBGR = cv2.imread(args["image"])
k = float(args['k'])
gamma = float(args["gamma"])

# Valores para ajustar el contraste
alow = float(args['low'])
ahigh = float(args['high'])

# Valores del nuevo rango de contraste
amin = float(args['min'])
amax = float(args['max'])

# Valores para el suavizado
blurtype = int(args["typeb"])
size = int(args["ksize"])

# Función para aplicar suavizado a la imagen
def smooth_image(image, kernel_size, blur_type):
    # Filtro Promedio (3x3 y 5x5)
    if blur_type == 0:
        if kernel_size == 3:
            average_kernel = np.ones((3, 3), np.float32) / 9.0
        elif kernel_size == 5:
            average_kernel = np.ones((5, 5), np.float32) / 25.0
        else:
            return image  # Si el kernel_size no es válido, devolver la imagen sin cambios
        output = cv2.filter2D(image, -1, average_kernel)
        return output

    # Filtro Gaussiano (3x3 y 5x5)
    elif blur_type == 1:
        if kernel_size == 3:
            gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        elif kernel_size == 5:
            gaussian_kernel = np.array([
                [1,  4,  6,  4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1,  4,  6,  4, 1]
            ]) / 256.0
        else:
            return image
        output = cv2.filter2D(image, -1, gaussian_kernel)
        return output

    # Filtro de Mediana (3x3 y 5x5)
    elif blur_type == 2:
        pad_size = kernel_size // 2
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        output = np.zeros_like(image)

        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                for c in range(image.shape[2]):  # Para cada canal de color
                    window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]
                    median_value = np.median(window)
                    output[i - pad_size, j - pad_size, c] = median_value
        return output

    # Filtro de Mediana Ponderada (3x3 y 5x5)
    elif blur_type == 4:
        pad_size = kernel_size // 2
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        output = np.zeros_like(image)

        # Kernel de pesos gaussiano para 3x3 y 5x5
        if kernel_size == 3:
            gaussian_weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        elif kernel_size == 5:
            gaussian_weights = np.array([[1,  4,  6,  4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1,  4,  6,  4, 1]])
        else:
            return image

        # Aplicación de la ventana deslizante
        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                for c in range(image.shape[2]):  # Para cada canal de color
                    window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]
                    weighted_values = []

                    # Expandir valores según los pesos del kernel
                    for wi in range(kernel_size):
                        for wj in range(kernel_size):
                            weight = gaussian_weights[wi, wj]
                            pixel_value = window[wi, wj]
                            weighted_values.extend([pixel_value] * int(weight))
                    
                    # Calcular la mediana ponderada
                    median_value = np.median(weighted_values)
                    output[i - pad_size, j - pad_size, c] = median_value
        return output

    # Filtro Bilateral
    elif blur_type == 3:
        return cv2.bilateralFilter(image, kernel_size, 21, 21)

    # Devolver la imagen sin cambios si no se cumple ninguna condición
    else:
        return image

#Función para aplicar la correccion gamma
def gamma_correction(b,g,r, gamma):
    
    # Aplicar la corrección gamma a cada canal
    b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
    g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
    r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
    
    # Combinar los canales corregidos de nuevo en una imagen BGR
    gamma_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    return gamma_corrected


# Función para ecualizar el histograma en formato HSV
def equalize_histogram_hsv(frame, k):
    # Convertir la imagen de BGR a HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Separamos los canales de la imagen
    h, s, v = cv2.split(image_HSV)
    
    # Calcular histograma
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    
    # Obtenemos las dimensiones de la imagen
    (M, N) = v.shape
    
    # Factor de cambio
    dx = (k - 1) / (M * N)
    
    # Construimos un vector Y para almacenar los valores precalculados
    y2 = np.array([np.round(cumulative_hist[i] * dx) for i in range(256)], dtype='uint8')
    
    # Aplicar la ecualización al canal V
    v_equalized = y2[v]
    
    image_HSV = cv2.merge([h, s, v_equalized])
    
    # Convertir la imagen de HSV de vuelta a BGR
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result

# Función para calcular el auto contraste restringido en formato HSV
def ajustar_contraste_hsv(frame, alow, ahigh, amin, amax):

    # Convertir la imagen de BGR a HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Trabajar con el canal V (Value)
    h, s, v = cv2.split(image_HSV)
    
    # Calcular histograma
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    
    # Obtenemos las dimensiones de la imagen
    (M, N) = v.shape
    
    # Obtenemos los valores de las condiciones para a'low y a'high
    multlow = int(M * N * alow)
    multhigh = int(M * N * (1 - ahigh))
    
    # Obtenemos a'low y a'high  (Rango de contraste restringido)
    alowp = min([i for i in range(256) if cumulative_hist[i] >= multlow])
    ahighp = max([i for i in range(256) if cumulative_hist[i] <= multhigh])
    
    # Factor de cambio
    dx = (amax - amin) / (ahighp - alowp)
    
    # Crear una tabla de mapeo con valores ajustados
    table_map = np.array([amin if i <= alowp else amax if i >= ahighp else amin + ((i - alowp) * dx) for i in range(256)], dtype='uint8')
    
    # Aplicar el mapeo al canal V
    v_correct = table_map[v]
    
    # Reemplazar el canal V ajustado en la imagen HSV
    image_HSV = cv2.merge([h, s, v_correct])
    
    # Convertir la imagen de HSV de vuelta a BGR
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result

# main
imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2GRAY)

# Mostrar imagen original
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1)
plt.imshow(imageRGB)
plt.title("Video original")
plt.axis('off')

# Separemos los canales de la imagen
b, g, r = cv2.split(imageBGR)

# Aplicar la corrección gamma
gamma_frame = gamma_correction(b, g, r, gamma)        
gamma_frame_rgb = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 2)
plt.imshow(gamma_frame_rgb)
plt.title("Video con corrección gamma")
plt.axis('off')

# Aplicar suavizado y mostrar
smooth = smooth_image(gamma_frame, size, blurtype)
smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 6)
plt.imshow(smooth_rgb)
plt.title("Video suavizado")
plt.axis('off')


# Aplicar y mostrar la ecualización del histograma
equalized_image = equalize_histogram_hsv(smooth, k)
equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 3)
plt.imshow(equalized_image_rgb)
plt.title("Video ecualizado")
plt.axis('off')

# Aplicar el ajuste de contraste y mostrar
image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
image_contrast_rgb = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 4)
plt.imshow(image_contrast_rgb)
plt.title("Video contrastado")
plt.axis('off')

# Convertir la imagen suavisada a escala de grises y mostrar
gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 3, 5)
plt.imshow(gray_frame, cmap="gray")
plt.title("Video final en gris")
plt.axis('off')

# Guardar la imagen procesada
cv2.imwrite(args["output"], gray_frame)

# Mostrar todas las imágenes en una única ventana
plt.tight_layout()
plt.show()
# cv2.imshow('Video original', imageBGR)

# # Aplicar suavizado a la imagen
# # image_smooth = smooth_image(imageBGR, size, blurtype)

# # sepamos los canales de la imagen
# # b, g, r = cv2.split(image_smooth)
# b, g, r = cv2.split(imageBGR)


# # Aplicar la corrección gamma
# gamma_frame = gamma_correction(b,g,r, gamma)        
# cv2.imshow('Video con correccion gamma', gamma_frame)
# #cv2.waitKey(0)
# # Aplicamos y mostramos la ecualización del histograma
# equalized_image = equalize_histogram_hsv(gamma_frame, k)        
# cv2.imshow('Video ecualizado', equalized_image)
# #cv2.waitKey(0)
# # Aplicamos el auto contraste restringido
# image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
# cv2.imshow('Video contrastado', image_contrast)

# # Convertir el video contrastado a escala de grises
# gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)

# smooth = smooth_image(image_contrast, size, blurtype)

# # Aplicamos y mostramos la mascara al video
# # masked_frame = mascara(gray_frame)
# cv2.imshow('Video final', gray_frame)
# cv2.waitKey(0)
# #Guadamos la imagen procesada
# # cv2.imwrite(args["output"], gray_frame)
# cv2.imwrite(args["output"], smooth)
# #output_video.write(masked_frame)




