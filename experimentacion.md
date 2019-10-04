# Experimentacion

Planeamos que cosas cabe experimentar.

1. Encontrar la mejor configuracion del algoritmo KNN.
2. Encontrar para esta configuracion de KNN, la mejor configuracion de PCA.
3. Encontrar si existe un configuracion suboptima de knn que funcione mejor con la mejor configuracion de PCA.
4. Encontrar la configuracion optima de ambas.

## Variables
Las variables disponibles para configurar los algoritmos son:
En knn:
- *neighbors*: El numero que recibe KNN en inicializacion. Define cuantos vecinos más cercanos al vector se consideran en la votacion.
- La norma usada para medir la distancia entre vectores `src/knn.cpp +46`. Por defecto, norma 2. Probar norma 1 y norma infinito.
- (Nice to have) Probar asignar peso a cada vecino basado en su distancia a la fila que se esta prediciendo.
- (Nice to have) Variar el algoritmo para determinar cuales son los vecinos mas cercanos.

En pca:
- *alpha*: El numero que recibe PCA en inicializacion.(Tengo entendido que) Define cuantas componentes sobreviven al algoritmo. 


## Extra
(Del taller) ¿Cómo cambia la performance si usamos menos instancias de entrenamiento? linealmente exponencialmente etc.
