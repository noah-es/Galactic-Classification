import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import tensorflow_datasets as tfds
import matplotlib.colors as colors
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import tensorflow_hub as hub
import os


"""## Las imágenes están todas en una única carpeta, así como los dos csv's de los que se parte para comenzar el proceso.
## El archivo ZIP que envío está distribuido de la siguiente forma.
## EL ZIP posee tres carpetas:
- Images: carpeta con todas las imágenes del dataset NA10.
- CSV'S: carpeta con los dos csv's que contienen la información de a qué imágen le corresponde cada clase
- Código: carpeta con el código de la práctica en cuestión.

- Esta organización está hecha a conciencia para que los paths relativos funcionen correctamente en cualquier ordenador que descargue y ejecute el código proveniente de la carpeta ZIP. Es importante no alterar la jerarquía del ZIP.

##__Análisis exploratorio de los datos__

## Carga de los CSV'S para su estudio
"""

# Los datos los proporciono yo en el fichero ZIP, tanto imágenes como csv's con las clasificaciones
# Aquí muestro el código sobre cómo a partir de dos csv's genero uno nuevo con los paths de las imágenes y sus respectivas clases

# Cargar los archivos CSV
df1 = pd.read_csv("../CSV's/NairAbrahamMorphology.csv", sep=';')
df2 = pd.read_csv("../CSV's/fcP_ID.csv", sep=';')

# Fusionar los DataFrames en función de la columna "spID"
nuevo_df = pd.merge(df1, df2, on="spID")

# Me quedo con las columnas que me interesan, el idendtificador del SDSS, el path y la clase
nuevo_df = nuevo_df[['#JID', 'fpCid', 'TType']]
# Guardar el nuevo DataFrame en un archivo CSV
nuevo_df.to_csv("../CSV's/path_TType.csv", index=False, sep=';')

def agregar_texto(celda):
    return f"../Images/{celda}.jpg"

# Aplicar la función a cada celda de la columna "fpCid"
nuevo_df["fpCid"] = nuevo_df["fpCid"].apply(agregar_texto)

# Guardar el DataFrame modificado en un nuevo archivo CSV
nuevo_df.to_csv("../CSV's/path_TType.csv", index=False, sep=';')

"""## Estructura del CSV generado"""

# Aquí muestro cómo es el nuevo csv generado.
# Además, muestro algunas imágenes del dataset.
nuevo_df

"""## Imágenes del dataset"""

# Visualización total o parcial de los datos

for i in nuevo_df['fpCid'].head(10):
    imagen = plt.imread(i)
    plt.imshow(imagen)
    plt.axis('off')  # Desactivar los ejes
    plt.show()

"""## Datos estadísticos del Dataset"""

# Obtención de estadísticas e información de los datos
# TODO (# datos por clase, estadísticos descriptivos, etc.)
clases = nuevo_df['TType'].value_counts()
total = nuevo_df['TType'].value_counts().sum()
print('------------------------------------------')
print('Distintas clases y número de imágenes por cada clase')
print(clases)
print('------------------------------------------')
print('Número total de imágenes')
print(total)
print('------------------------------------------')

"""## Histograma de las dimensiones de las imágenes"""

# Ahora voy a hacer un histograma para ver la distribución de dimensiones que hay en el catálogo

# 1. Empiezo haciendo una función que recoge en una lista las dimensiones de cada imagen

def dimension(ruta_carpeta):
    dimensiones = []

    for filename in os.listdir(ruta_carpeta):
        if filename.endswith(".jpg") or filename.endswith(".png"): # Asegúrate de que solo procese imágenes
            try:
                with Image.open(os.path.join(ruta_carpeta, filename)) as img:
                    width, height = img.size
                    dimensiones.append((width, height))
            except:
                print(f"No se pudo procesar la imagen {filename}")

    return dimensiones

# 2. Defino la ruta de la carpeta y le aplico la función
ruta_carpeta = "../Images"
dimensiones = dimension(ruta_carpeta)

# 3. Contar la frecuencia de cada tipo único de (ancho x alto)
dimensiones_dict = {}
for dimension in dimensiones:
    if dimension in dimensiones_dict:
        dimensiones_dict[dimension] += 1
    else:
        dimensiones_dict[dimension] = 1

# 4. Extraer los valores de (ancho x alto) y sus frecuencias y ordenar las dimensiones por tamaño
dimensiones_ordenadas = sorted(dimensiones_dict.items(), key=lambda x: x[0][0] * x[0][1])
dimensiones_valores = [str(dimension[0]) for dimension in dimensiones_ordenadas]
frecuencias = [dimension[1] for dimension in dimensiones_ordenadas]

# 5. Plotear el resultado
x = np.arange(len(dimensiones_valores))
plt.bar(x, frecuencias, color='blue')
plt.xlabel('Dimensiones (Ancho x Alto)')
plt.ylabel('Frecuencia')
plt.title('Histograma de Dimensiones de Imágenes')
# plt.xticks([])
# plt.xticks(x, dimensiones_valores, rotation='vertical')
# Seleccionar un subconjunto de dimensiones para mostrar como ticks en el eje x
num_ticks = 20
indices_ticks = np.round(np.linspace(0, len(dimensiones_valores) - 1, num_ticks)).astype(int)
dimensiones_valores_ticks = [dimensiones_valores[i] for i in indices_ticks]
plt.xticks(indices_ticks, dimensiones_valores_ticks, rotation='vertical')
plt.savefig("histograma.png", bbox_inches='tight')
plt.show()

"""## Minería de datos. Imágenes corruptas"""

# Ahora quiero ver si hay imágenes corruptas. Para ello compruebo que no haya cuadrantes en negro o por debajo de cierto umbral
ruta_carpeta = "../Images"
corrupta1 = []
# Lista de archivos en la carpeta
archivos = os.listdir(ruta_carpeta)

# Iterar sobre las imágenes en la carpeta
for nombre_archivo in os.listdir(ruta_carpeta):
    ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)

    # Leer el archivo de imagen
    contenido = tf.io.read_file(ruta_archivo)

    # Decodificar el contenido en un tensor de imagen
    imagen_tensor = tf.image.decode_image(contenido)

    # Obtener dimensiones de la imagen
    alto, ancho, _ = imagen_tensor.shape

    # Dividir la imagen en cuatro cuadrantes
    cuadrante_superior_izquierdo = imagen_tensor[:alto//2, :ancho//2, :]
    cuadrante_superior_derecho = imagen_tensor[:alto//2, ancho//2:, :]
    cuadrante_inferior_izquierdo = imagen_tensor[alto//2:, :ancho//2, :]
    cuadrante_inferior_derecho = imagen_tensor[alto//2:, ancho//2:, :]

    # Verificar si los píxeles en el segundo y cuarto cuadrante son negros
    if np.all(cuadrante_superior_izquierdo < 130) and np.all(cuadrante_inferior_izquierdo < 130):
        corrupta1.append(nombre_archivo)

# Imprimir las imágenes parcialmente negras encontradas
print("Imágenes parcialmente negras:")
for nombre_archivo in corrupta1:
    imagen = plt.imread(os.path.join(ruta_carpeta, nombre_archivo))
    # Mostrar la imagen
    plt.imshow(imagen)
    plt.axis('off')  # Desactiva los ejes
    plt.show()

# Finalmente estas imágenes las borro del dataset, porque son inservibles
nueva_lista = ['../Images/fpC-002738-40-5-0216-0087.jpg', '../Images/fpC-002247-41-4-0285-0037.jpg', '../Images/fpC-002247-41-4-0285-0076.jpg', '../Images/fpC-002131-40-4-0054-0191.jpg', '../Images/fpC-000752-40-2-0452-0063.jpg', '../Images/fpC-000756-44-2-0637-0103.jpg', '../Images/fpC-001035-40-4-0183-0107.jpg']

# for elemento in corrupta1:
#   nuevo_elemento = f"../Images/{elemento}"
#   nueva_lista.append(nuevo_elemento)

# print(nueva_lista)

df_filtrado = nuevo_df[~nuevo_df['fpCid'].isin(nueva_lista)]

df_filtrado.to_csv("../CSV's/path_TType.csv", index=False, sep=';')

df_filtrado.describe()

"""## Reajuste del dataset
- Pretendo trabajar con 3 clases. Las clases que vimos anteriormente son subclases dentro de clases de orden superior.
- Aquí lo que hago es crear otra columna que recoja la información de imagen-clase de orden superior.
"""

# Ahora creo otra columna de clase_final que agrupará todas las sub-clases en 4 clases principales
def assign_final_class(clase):
    if clase == -5:
        return 'Ellipticals'
    elif -3 <= clase <= 0:
        return 'S0s'
    elif 1 <= clase <= 9:
        return 'Spirals'
    else:
        return 'Irr-Misc'  # Si la clase no cumple con ninguna de las condiciones anteriores, puedes asignar otro valor

# Crear la nueva columna "Final_Class" basada en la columna de clase original
df_filtrado['Clase'] = df_filtrado['TType'].apply(assign_final_class)
df_filtrado['Clase'].value_counts()

"""##__Creación del data set en TensorFlow__

## Genero los conjuntos de datos en CSV'S distintos
"""

# Ahora la idea es meter en val y test 300 imágenes de cada clase, pero aleatoriamente.

# Creo DataFrames vacíos para train, val y test
train_df = pd.DataFrame(columns=['#JID', 'fpCid', 'TType', 'Clase'])
val_df = pd.DataFrame(columns=['#JID', 'fpCid', 'TType', 'Clase'])
test_df = pd.DataFrame(columns=['#JID', 'fpCid', 'TType', 'Clase'])

# Definir el número de imágenes por clase en val y test
num_images_per_class = 300

# Iterar sobre cada clase
for clase in df_filtrado['Clase'].unique():
    # Seleccionar las filas correspondientes a la clase actual
    clase_df = df_filtrado[df_filtrado['Clase'] == clase]

    # Barajar las filas aleatoriamente
    clase_df = clase_df.sample(frac=1).reset_index(drop=True)

    # Dividir las imágenes de la clase en val y test
    val_and_test = clase_df.head(num_images_per_class * 2)
    val_df = pd.concat([val_df, val_and_test.head(num_images_per_class)])
    test_df = pd.concat([test_df, val_and_test.tail(num_images_per_class)])

    # Las imágenes restantes van al conjunto de entrenamiento
    train_df = pd.concat([train_df, clase_df.iloc[num_images_per_class * 2:]])

# Guardar los DataFrames en archivos CSV
train_df.to_csv("../CSV's/Train.csv", index=False, sep=';')
val_df.to_csv("../CSV's/Val.csv", index=False, sep=';')
test_df.to_csv("../CSV's/Test.csv", index=False, sep=';')

# Como IRR+MISC posee muy pocas instancias, primero vamos a realizar un análisis de tres clases: Elip, Spirals y S0s
# Para ello, debo eliminar del val y test las instancias de Irr+Misc
val_df = val_df[val_df['Clase'] != 'Irr-Misc']
test_df = test_df[test_df['Clase'] != 'Irr-Misc']

val_df.to_csv("../CSV's/Val1.csv", index=False, sep=';')
test_df.to_csv("../CSV's/Test1.csv", index=False, sep=';')

"""## Función aumento de datos"""

# Estudio de técnicas de aumento de datos
def mi_funcion_python_min(img, label):

    img = img.numpy().reshape(224,224,3)

    # Creo el número aleatorio
    numero_aleatorio = random.uniform(0, 0.4)

    # Ahora controlo qué aumento se hace en  función de ese número
    if numero_aleatorio < 0.1:

        # 0 means flipping around the x-axis
        # 1 means flipping around y-axis
        # -1 means flipping around both axes.
        aleat = random.randint(-1, 1)
        augmented = cv2.flip(img, aleat)

    elif numero_aleatorio < 0.2:

        aleat = random.randint(0,360)
        alto, ancho = img.shape[:2]
        centro = (ancho // 2, alto // 2)
        matriz_rotacion = cv2.getRotationMatrix2D(centro, aleat, 1.0)
        augmented = cv2.warpAffine(img, matriz_rotacion, (ancho, alto), flags=cv2.INTER_LINEAR)

    # elif numero_aleatorio < 0.3:

        # aleat = random.randint(0,1)
        # aleat = aleat if aleat % 2 != 0 else aleat + 1
        # augmented=np.copy(img)
        # augmented[:, :, 0]= cv2.GaussianBlur(img[:, :, 0], (aleat,aleat), 0)  # Canal R
        # augmented[:, :, 1] = cv2.GaussianBlur(img[:, :, 1], (aleat,aleat), 0)  # Canal G
        # augmented[:, :, 2] = cv2.GaussianBlur(img[:, :, 2], (aleat,aleat), 0)  # Canal B

    # elif numero_aleatorio < 0.4:

        # a = random.randint(0,1)
        # b = random.randint(0,1)
        # kernel = np.ones((a,b),np.uint8)
        # augmented = cv2.dilate(img,kernel,iterations = 1)

    # elif numero_aleatorio < 0.5:

        # a = random.uniform(5,5.1)
        # kernel = np.array([
        #                   [-1, -1, -1],
        #                   [-1, 9, -1],
        #                   [-1, -1, -1]
        #                             ])
        # augmented = cv2.filter2D(img, -1, kernel)

    elif numero_aleatorio < 0.3:

        a = random.uniform(-0.3,0.3)
        b = random.uniform(-0.3,0.3)
        alto, ancho = img.shape[:2]
        M = np.float32([[1, a, 0],   # No hay cambio en x, un cizallamiento de 0.5 en y
                       [b, 1, 0]])
        augmented = cv2.warpAffine(img, M, (ancho, alto))

    elif numero_aleatorio < 0.4:

        a = random.randint(-15,15)
        b = random.randint(-15,15)
        M_traslacion = np.float32 ([[1, 0, a],
                                   [0, 1, b]])
        alto, ancho = img.shape[:2]
        augmented = cv2.warpAffine(img, M_traslacion, (ancho, alto))

    # elif numero_aleatorio <= 0.7:

        # a = random.uniform(0,0.01)
        # ruido = np.random.normal(0, a, img.shape).astype(np.float32)
        # augmented = cv2.add(img, ruido)


    return augmented, label

# Estudio de técnicas de aumento de datos
def mi_funcion_python_max(img, label):

    img = img.numpy().reshape(224,224,3)

    # Creo el número aleatorio
    numero_aleatorio = random.uniform(0, 0.8)

    # Ahora controlo qué aumento se hace en  función de ese número
    if numero_aleatorio < 0.1:

        # 0 means flipping around the x-axis
        # 1 means flipping around y-axis
        # -1 means flipping around both axes.
        aleat = random.randint(-1, 1)
        augmented = cv2.flip(img, aleat)

    elif numero_aleatorio < 0.2:

        aleat = random.randint(0,360)
        alto, ancho = img.shape[:2]
        centro = (ancho // 2, alto // 2)
        matriz_rotacion = cv2.getRotationMatrix2D(centro, aleat, 1.0)
        augmented = cv2.warpAffine(img, matriz_rotacion, (ancho, alto), flags=cv2.INTER_LINEAR)

    elif numero_aleatorio < 0.3:

        aleat = random.randint(0,1)
        aleat = aleat if aleat % 2 != 0 else aleat + 1
        augmented=np.copy(img)
        augmented[:, :, 0]= cv2.GaussianBlur(img[:, :, 0], (aleat,aleat), 0)  # Canal R
        augmented[:, :, 1] = cv2.GaussianBlur(img[:, :, 1], (aleat,aleat), 0)  # Canal G
        augmented[:, :, 2] = cv2.GaussianBlur(img[:, :, 2], (aleat,aleat), 0)  # Canal B

    elif numero_aleatorio < 0.4:

        a = random.randint(0,1)
        b = random.randint(0,1)
        kernel = np.ones((a,b),np.uint8)
        augmented = cv2.dilate(img,kernel,iterations = 1)

    elif numero_aleatorio < 0.5:

        a = random.uniform(5,5.1)
        kernel = np.array([
                          [-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]
                                    ])
        augmented = cv2.filter2D(img, -1, kernel)

    elif numero_aleatorio < 0.6:

        a = random.uniform(-0.3,0.3)
        b = random.uniform(-0.3,0.3)
        alto, ancho = img.shape[:2]
        M = np.float32([[1, a, 0],   # No hay cambio en x, un cizallamiento de 0.5 en y
                       [b, 1, 0]])
        augmented = cv2.warpAffine(img, M, (ancho, alto))

    elif numero_aleatorio < 0.7:

        a = random.randint(-15,15)
        b = random.randint(-15,15)
        M_traslacion = np.float32 ([[1, 0, a],
                                   [0, 1, b]])
        alto, ancho = img.shape[:2]
        augmented = cv2.warpAffine(img, M_traslacion, (ancho, alto))

    elif numero_aleatorio <= 0.8:

        a = random.uniform(0,0.01)
        ruido = np.random.normal(0, a, img.shape).astype(np.float32)
        augmented = cv2.add(img, ruido)


    return augmented, label

"""## Decorador TF"""

# Envuelve tu función Python con tf.py_function
@tf.function
def aumento_min(arg1, arg2):
    return tf.py_function(mi_funcion_python_min, [arg1, arg2], [tf.float32, tf.int64])

# Envuelve tu función Python con tf.py_function
@tf.function
def aumento_max(arg1, arg2):
    return tf.py_function(mi_funcion_python_max, [arg1, arg2], [tf.float32, tf.int64])

"""### Proceso de creación del dataset. Cargar, normalizar, reescalar y preparar los dataset para entrenar.
- Abro las imagenes con tf y le indico que están a color, RGB.
- Aplico un cast para pasar la imagen a float32 y  normalizo los pixeles entre _[0,1]_.
- Hago un recorte quedándome con el 50% central de la imagen y cambio el número de pixeles a (100,100).
- Defino una función auxiliar que me permite seleccionar cualquiero canal de la imagen para entrenar.
- Defino una función que me permite, a partir de las listas del dataset, tomar aleatoriamente un número concreto de imágenes para cada clase.

"""

def cargar(imagen,label):
    image_bytes = tf.io.read_file(imagen)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    return image, label

def normalizar(imagen,label):
    image = tf.cast(imagen, tf.float32)
    image /= 255.
    return image, label

def resize(imagen,label):
    imagen = tf.image.central_crop(imagen, 0.5)
    imagen = tf.image.resize(imagen, (224,224))
    return imagen,label

def b_band(imagen, label):
    return imagen[:,:,1], label

"""## Funcion Prepare. Es la función más importante del código. Posee flags que me permiten aplicar tratamientos distintos a Train.
- Abre el csv del dataset que quiero preparar y cojo las columnas de 'fpCid' y 'Clase', que recogen los paths relativos de las imágenes y sus clases, respectivamente.
- A partir de las dos listas obtengo del CSV, uso tf.data.Dataset.from_tensor_slices para generar mi dataset a partir de dos listas.
- Con .map aplico las funciones anteriores al dataset.
"""

def prepare(ruta, shuffle=False, train=False, max = False, min = False):

    BUFFER = 200
    LOTE = 200
    MUESTRAS_POR_CLASE=2100

    # Abro los datos con Pandas
    data = pd.read_csv(ruta, sep = ';', index_col='#JID')

    # Guardo en una lista los paths relativos y sus clases, codificadas a numeros
    file_paths = data['fpCid'].values.tolist()
    mappings = {'Spirals': 0, 'S0s': 1, 'Ellipticals': 2}
    data['Clase'] = data['Clase'].map(mappings)
    labels = data['Clase'].values.tolist()

    # Ahora yo quiero que cada vez que entrene, se cojan aleatoriamente de cada
    # clase 2000 instancias, de tal forma que elimino el desbalanceo de clases
    def select_samples(file_paths, labels):
    # Convertir a numpy arrays
        file_paths_np = np.array(file_paths)
        labels_np = np.array(labels)
        selected_indices = []
        for clase in np.unique(labels_np):
            # Indices de muestras de la clase actual
            indices_clase = np.where(labels_np == clase)[0]
            # Seleccionar muestras aleatorias
            selected_indices.extend(np.random.choice(indices_clase, MUESTRAS_POR_CLASE, replace=False))
        return file_paths_np[selected_indices], labels_np[selected_indices]
    # ------------------------------------------------------
    # TODO ESTO OCURRIRÁ CUANDO EL MODELO LLAME A TRAIN_DS -
    # ------------------------------------------------------

    # Crear el dataset a partir de las listas. Ahora tengo un dataset, pero con la diferencia de que mi X_train es un vector de paths.
    # Esto me sirve para hacer operaciones con los datos sin tener que manejar tensores. Más rápido.

    # Creo mi dataset a partir de dos listas
    if train:
        file_paths, labels = select_samples(file_paths, labels)
    valores_unicos, frecuencias = np.unique(labels, return_counts=True)
    print(frecuencias)
    train_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Ahora quiero barajarlo. Pongo un flag para decidir si quiero o no quiero barajar.
    if shuffle:
        train_ds = train_ds.shuffle(BUFFER)

    # Ahora quiero cargar las imágenes en tensorflow. Aplico la función cargar.
    train_ds = train_ds.map(cargar, tf.data.experimental.AUTOTUNE)

    # Mormalizo los píxeles
    train_ds = train_ds.map(normalizar, tf.data.experimental.AUTOTUNE)

    # Hago un resize para equiparar todas las imágenes.
    train_ds = train_ds.map(resize, tf.data.experimental.AUTOTUNE)

    # Ahora voy a probar a transformar el dataset en imágenes en la banda b
    # train_ds = train_ds.map(b_band, tf.data.experimental.AUTOTUNE)

    # Muestro por primera vez la forma del dataset. Es una medida de control
    print(train_ds)

    # Aplico la función de aumento de datos. Vuelvo a aplicar un flag para tener control sobre el mismo.
    if train:
        if min:
          # Aplica la función a los datos
          train_ds = train_ds.map(aumento_min)
          # Asigna formas fijas a los tensores de salida
          train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x, [224,224,3]), tf.ensure_shape(y, [])))
        elif max:
          # Aplica la función a los datos
          train_ds = train_ds.map(aumento_max)
          # Asigna formas fijas a los tensores de salida
          train_ds = train_ds.map(lambda x, y: (tf.ensure_shape(x, [224,224,3]), tf.ensure_shape(y, [])))

    # Vuelvo a mostar la forma del dataset
    print(train_ds)

    # Guardo en cache en vez del disco para agilizar el proceso
    train_ds = train_ds.cache()

    # Los guardo en lotes
    train_ds = train_ds.batch(LOTE)

    # Lo prefetcheo
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Pinto por última vez las caracteristicas del dataset
    print(train_ds)
    print('hola')
    return train_ds

"""## Visualizo y compruebo que el dataset está correcto y que el aumento de datos funciona."""

# Código para visualizar y comprobar que la lectura de los datos por parte de los dataset es correcta
train_ds = prepare("../CSV's/Train.csv", shuffle=True, train=True, min = True, max = False)

for elemento, label in train_ds.take(1):
        plt.imshow(elemento[0,:,:].numpy())
plt.show()

val_ds = prepare("../CSV's/Val1.csv", shuffle=False, train=False, min = False, max = False)

test_ds = prepare("../CSV's/Test1.csv", shuffle=False, train=False, min = False, max = False)

"""##__Definición de la función de pérdida y las métricas de evaluación__"""

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
metricas = ['accuracy']

"""## Mi learning Rate seguirá un decaimiento exponencial"""

lr_schedule = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=32,
    decay_rate=0.9894)
optimizador = tf.optimizers.Adam(lr_schedule)

"""## Genero dos callbacks. Uno que me va guardando el mejor modelo y otro con paciencia 7."""

# Código para definir los hiperparámetros del entrenamiento
EPOCAS = 200
early_stopping = EarlyStopping(monitor='val_loss', patience=7)

checkpoint_filepath = '../Modelo/MobileNetV2.keras'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

"""## Veo que para redes neuronales densas, la clasificación de imágenes es áltamente difícil.

## Voy a usar transfer learning.

## MobilenetV2
"""

IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()

# Ahora creo mi modelo

transfer = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(3)
])

transfer.summary()

tf.keras.utils.plot_model(transfer, show_shapes=True)

transfer.compile(
    optimizer = optimizador,
    loss=loss,
    metrics= metricas,
)

history1 = transfer.fit(train_ds, epochs=EPOCAS, validation_data=val_ds, callbacks = [early_stopping, checkpoint_callback])

checkpoint_filepath2 = '../Modelo/MobileNetV2_FineTuning.keras'
checkpoint_callback2 = ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

base_model.trainable = True

# Número total de capas en el modelo base
total_layers = len(base_model.layers)

# Definir el punto de corte para las últimas 35 capas
fine_tune_at = total_layers - 20

# Congelar todas las capas antes de la capa `fine_tune_at`
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Asegurarse de que las capas de Batch Normalization permanezcan congeladas
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

optimizador2 = tf.optimizers.Adam(0.000007)

transfer.compile(
    optimizer=optimizador2,
    loss=loss,
    metrics=metricas,
)

# Entrenar el modelo nuevamente
history2 = transfer.fit(train_ds, initial_epoch=200, epochs=len(history1.history['loss']) + 100, validation_data=val_ds, callbacks=[checkpoint_callback2])

# Código para visualizar y comprobar que la lectura de los datos por parte de los dataset es correcta
train_ds = prepare("../CSV's/Train.csv", shuffle=True, train=True, min = False, max = True)

optimizador2 = tf.optimizers.Adam(0.000007)

transfer.compile(
    optimizer=optimizador2,
    loss=loss,
    metrics=metricas,
)

# Entrenar el modelo nuevamente
history3 = transfer.fit(train_ds, initial_epoch=300, epochs=100, validation_data=val_ds, callbacks=[checkpoint_callback2])

def combinar_histories(history1, history2):
    history_combined = {}
    for key in history1.history.keys():
        history_combined[key] = history1.history[key] + history2.history[key]
    return history_combined

history_combined = combinar_histories(history_combined, history3)

plt.figure()
plt.plot(history_combined['loss'][:294], '--', label='Loss (training data)')
plt.plot(history_combined['val_loss'][:294], label='Loss (validation data)')
plt.ylim([0, 3.5])
plt.plot([200-1,200-1],
          plt.ylim(), label='Start Fine Tuning')
plt.title('Loss para MobileNetV2')
plt.ylabel('CE')
plt.xlabel('Nº epoch')
plt.legend(loc="upper right")
plt.show()

# Plot history: Accuracy
plt.figure()
plt.plot(history_combined['accuracy'][:294], '--', label='Accuracy (training data)')
plt.plot(history_combined['val_accuracy'][:294], label='Accuracy (validation data)')
plt.ylim([0, 1])
plt.plot([200-1,200-1],
          plt.ylim(), label='Start Fine Tuning')
plt.title('Accuracy para MobileNetV2')
plt.ylabel('Accuracy')
plt.xlabel('Nº epoch')
plt.legend(loc="lower right")
plt.show()

step = np.linspace(0,200*32)

lr = lr_schedule(step)
plt.figure()
plt.plot(step/32, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

val_ds = prepare("../CSV's/Val1.csv", shuffle=True, train=False, min = False, max = False)

from tensorflow.keras.models import load_model
modelo = load_model('../Modelo/MobileNetV2.keras', custom_objects={'KerasLayer': hub.KerasLayer})

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import tensorflow as tf

# Obtener las predicciones

Y_pred = modelo.predict(test_ds)
def softmax(logits):  # para multiclase
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

Y_pred = softmax(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)

# Verdaderos valores
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Crear un heatmap con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=['Spirals', 'S0s', 'Ellipticals'], yticklabels=['Spirals', 'S0s', 'Ellipticals'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Reporte de clasificación
class_report = classification_report(y_true, y_pred, target_names=['Spirals', 'S0s', 'Ellipticals'])
print('Classification Report')
print(class_report)

# Calcular métricas para cada clase
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)  # Recall es lo mismo que sensibilidad
f1_per_class = f1_score(y_true, y_pred, average=None)
specificity_per_class = []

for i in range(len(['Spirals', 'S0s', 'Ellipticals'])):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - conf_matrix[i, i]
    tp = conf_matrix[i, i]
    specificity_per_class.append(tn / (tn + fp))

# Calcular promedios ponderados
weighted_precision = precision_score(y_true, y_pred, average='weighted')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
weighted_specificity = np.average(specificity_per_class, weights=np.bincount(y_true))

# Mostrar las métricas por clase
for i, class_name in enumerate(['Spirals', 'S0s', 'Ellipticals']):
    print(f'Class: {class_name}')
    print(f'  Precision: {precision_per_class[i]}')
    print(f'  Sensitivity (Recall): {recall_per_class[i]}')
    print(f'  Specificity: {specificity_per_class[i]}')
    print(f'  F1 Score: {f1_per_class[i]}')

# Mostrar los promedios ponderados
print(f'Weighted Precision: {weighted_precision}')
print(f'Weighted Sensitivity (Recall): {weighted_recall}')
print(f'Weighted Specificity: {weighted_specificity}')
print(f'Weighted F1 Score: {weighted_f1}')

# Curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(['Spirals', 'S0s', 'Ellipticals'])

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar todas las curvas ROC
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve (class {0}) (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Definir el número de imágenes a mostrar
class_names = ['Spirals', 'S0s', 'Ellipticals']
images = np.concatenate([x for x, y in test_ds], axis=0)

# Crear la figura
plt.figure(figsize=(15, 15))
contador = 0
for i in range(1,len(images),80):
    plt.subplot(3, 4, contador + 1)
    plt.imshow(images[i])
    plt.title(f'Real: {class_names[y_true[i]]}, Pred: {class_names[y_pred[i]]} ({np.max(Y_pred[i]):.2f})')
    plt.axis('off')
    contador += 1

plt.tight_layout()
plt.show()

len(images)

"""## Efficientnet"""

url = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b0-feature-vector/1"
efficientnetv2 = hub.KerasLayer(url, input_shape = (224,224,3), trainable=False)

# Ahora creo mi modelo

transfer0 = tf.keras.Sequential([
    efficientnetv2,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(3)
])

transfer0.summary()

tf.keras.utils.plot_model(transfer0, show_shapes=True)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
metricas = ['accuracy']

lr_schedule = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=32,
    decay_rate=0.9894)
optimizador = tf.optimizers.Adam(lr_schedule)

# Código para definir los hiperparámetros del entrenamiento
EPOCAS = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=7)

transfer0.compile(
    optimizer = optimizador,
    loss=loss,
    metrics= metricas,
)

checkpoint_filepath0 = '../Modelo/mejor_modelo_efficient1.keras'
checkpoint_callback0 = ModelCheckpoint(
    filepath=checkpoint_filepath0,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

history3 = transfer0.fit(train_ds, epochs=300, validation_data=val_ds, callbacks = [early_stopping, checkpoint_callback0])

plt.figure()
plt.plot(history3.history['loss'], '--', label='Train Loss')
plt.plot(history3.history['val_loss'],   label='Validation Loss')
plt.ylim([0.2, 2.5])
plt.title('Loss para EficcientNetV2')
plt.ylabel('CE')
plt.xlabel('Nº epoch')
plt.legend(loc="upper right")
plt.show()

# Plot history: Accuracy
plt.figure()
plt.plot(history3.history['accuracy'], '--', label='Train Accuracy')
plt.plot(history3.history['val_accuracy'], label='Validation Accuracy')
plt.ylim([0.1, 0.9])
plt.title('Accuracy para EficcientNetV2')
plt.ylabel('Accuracy')
plt.xlabel('Nº epoch')
plt.legend(loc="lower right")
plt.show()

step = np.linspace(0,300*32)

lr = lr_schedule(step)
plt.figure()
plt.plot(step/32, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

from tensorflow.keras.models import load_model
modelo = load_model('../Modelo/mejor_modelo_efficient1.keras', custom_objects={'KerasLayer': hub.KerasLayer})

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import tensorflow as tf

# Obtener las predicciones

Y_pred = modelo.predict(test_ds)
def softmax(logits):  # para multiclase
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

Y_pred = softmax(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)

# Verdaderos valores
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Crear un heatmap con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=['Spirals', 'S0s', 'Ellipticals'], yticklabels=['Spirals', 'S0s', 'Ellipticals'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Reporte de clasificación
class_report = classification_report(y_true, y_pred, target_names=['Spirals', 'S0s', 'Ellipticals'])
print('Classification Report')
print(class_report)

# Calcular métricas para cada clase
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)  # Recall es lo mismo que sensibilidad
f1_per_class = f1_score(y_true, y_pred, average=None)
specificity_per_class = []

for i in range(len(['Spirals', 'S0s', 'Ellipticals'])):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - conf_matrix[i, i]
    tp = conf_matrix[i, i]
    specificity_per_class.append(tn / (tn + fp))

# Calcular promedios ponderados
weighted_precision = precision_score(y_true, y_pred, average='weighted')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
weighted_specificity = np.average(specificity_per_class, weights=np.bincount(y_true))

# Mostrar las métricas por clase
for i, class_name in enumerate(['Spirals', 'S0s', 'Ellipticals']):
    print(f'Class: {class_name}')
    print(f'  Precision: {precision_per_class[i]}')
    print(f'  Sensitivity (Recall): {recall_per_class[i]}')
    print(f'  Specificity: {specificity_per_class[i]}')
    print(f'  F1 Score: {f1_per_class[i]}')

# Mostrar los promedios ponderados
print(f'Weighted Precision: {weighted_precision}')
print(f'Weighted Sensitivity (Recall): {weighted_recall}')
print(f'Weighted Specificity: {weighted_specificity}')
print(f'Weighted F1 Score: {weighted_f1}')

# Curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(['Spirals', 'S0s', 'Ellipticals'])

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar todas las curvas ROC
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve (class {0}) (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Definir el número de imágenes a mostrar
class_names = ['Spirals', 'S0s', 'Ellipticals']
images = np.concatenate([x for x, y in test_ds], axis=0)

# Crear la figura
plt.figure(figsize=(15, 15))
contador = 0
for i in range(1,len(images),80):
    plt.subplot(3, 4, contador + 1)
    plt.imshow(images[i])
    plt.title(f'Real: {class_names[y_true[i]]}, Pred: {class_names[y_pred[i]]} ({np.max(Y_pred[i]):.2f})')
    plt.axis('off')
    contador += 1

plt.tight_layout()
plt.show()

"""ConvNeXt"""

url = "https://www.kaggle.com/models/spsayakpaul/convnext/tensorFlow2/tiny-1k-224-fe/1"
base_model = hub.KerasLayer(url, input_shape = (224,224,3), trainable=False)

# Ahora creo mi modelo

transfer = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(3)
])

transfer.summary()

tf.keras.utils.plot_model(transfer, show_shapes=True)

transfer.compile(
    optimizer = optimizador,
    loss=loss,
    metrics= metricas,
)

checkpoint_filepath00 = '../Modelo/CovNeXt.keras'
checkpoint_callback00 = ModelCheckpoint(
    filepath=checkpoint_filepath00,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

history1 = transfer.fit(train_ds, epochs=100, validation_data=val_ds, callbacks = [early_stopping, checkpoint_callback00])

plt.figure()
plt.plot(history1['loss'], '--', label='Loss (training data)')
plt.plot(history1['val_loss'], label='Loss (validation data)')
plt.title('Loss para CovNeXt')
plt.ylabel('CE')
plt.xlabel('Nº epoch')
plt.legend(loc="upper right")
plt.show()

# Plot history: Accuracy
plt.figure()
plt.plot(history1['accuracy'], '--', label='Accuracy (training data)')
plt.plot(history1['val_accuracy'], label='Accuracy (validation data)')
plt.title('Accuracy para CovNeXt')
plt.ylabel('Accuracy')
plt.xlabel('Nº epoch')
plt.legend(loc="lower right")
plt.show()

step = np.linspace(0,100*32)

lr = lr_schedule(step)
plt.figure()
plt.plot(step/32, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

from tensorflow.keras.models import load_model
modelo2 = load_model('../Modelo/CovNeXt.keras', custom_objects={'KerasLayer': hub.KerasLayer})

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import tensorflow as tf

# Obtener las predicciones

Y_pred = modelo2.predict(test_ds)
def softmax(logits):  # para multiclase
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

Y_pred = softmax(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)

# Verdaderos valores
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Crear un heatmap con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='Blues', xticklabels=['Spirals', 'S0s', 'Ellipticals'], yticklabels=['Spirals', 'S0s', 'Ellipticals'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Reporte de clasificación
class_report = classification_report(y_true, y_pred, target_names=['Spirals', 'S0s', 'Ellipticals'])
print('Classification Report')
print(class_report)

# Calcular métricas para cada clase
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)  # Recall es lo mismo que sensibilidad
f1_per_class = f1_score(y_true, y_pred, average=None)
specificity_per_class = []

for i in range(len(['Spirals', 'S0s', 'Ellipticals'])):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - conf_matrix[i, i]
    tp = conf_matrix[i, i]
    specificity_per_class.append(tn / (tn + fp))

# Calcular promedios ponderados
weighted_precision = precision_score(y_true, y_pred, average='weighted')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
weighted_specificity = np.average(specificity_per_class, weights=np.bincount(y_true))

# Mostrar las métricas por clase
for i, class_name in enumerate(['Spirals', 'S0s', 'Ellipticals']):
    print(f'Class: {class_name}')
    print(f'  Precision: {precision_per_class[i]}')
    print(f'  Sensitivity (Recall): {recall_per_class[i]}')
    print(f'  Specificity: {specificity_per_class[i]}')
    print(f'  F1 Score: {f1_per_class[i]}')

# Mostrar los promedios ponderados
print(f'Weighted Precision: {weighted_precision}')
print(f'Weighted Sensitivity (Recall): {weighted_recall}')
print(f'Weighted Specificity: {weighted_specificity}')
print(f'Weighted F1 Score: {weighted_f1}')

# Curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(['Spirals', 'S0s', 'Ellipticals'])

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar todas las curvas ROC
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve (class {0}) (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Definir el número de imágenes a mostrar
class_names = ['Spirals', 'S0s', 'Ellipticals']
images = np.concatenate([x for x, y in test_ds], axis=0)

# Crear la figura
plt.figure(figsize=(15, 15))
contador = 0
for i in range(1,len(images),80):
    plt.subplot(3, 4, contador + 1)
    plt.imshow(images[i])
    plt.title(f'Real: {class_names[y_true[i]]}, Pred: {class_names[y_pred[i]]} ({np.max(Y_pred[i]):.2f})')
    plt.axis('off')
    contador += 1

plt.tight_layout()
plt.show()

"""##__Inferencia del modelo__

# Cargar una red entrenada desde nuestro equipo
"""

from tensorflow.keras.models import load_model
modelo2 = load_model('../Modelo/mejor_modelo2.keras', custom_objects={'KerasLayer': hub.KerasLayer})

"""## Inferencia"""

test_ds = prepare("../CSV's/Test1.csv", shuffle=False, train=False)
predicciones = modelo2.predict(test_ds)

def softmax(logits):  # para multiclase
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

print("Predicciones sobre Test:", softmax(predicciones[0:5]))
test_df['Clase'].head(5)
