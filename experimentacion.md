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
- *alpha*: El numero que recibe PCA en inicializacion.(Tengo entendido que) Define cuantas componentes sobreviven al algoritmo. Variar entre 50 y 400
- Criterio de para del metodo de potencia

Reducir el training set es aceptable si se mantienen buen representadasa las clases.

Sacar palabras de las reseñas es posible, pero hay que hacer un buen analisis para justificarlo (sacar las palabras que son demasiado frecuentes y no aportan informacion)En este contexto palabras como articulos, pelicula y cine.

## Extra
(Del taller de knn) ¿Cómo cambia la performance si usamos menos instancias de entrenamiento? linealmente exponencialmente etc.


## Python del taller

Fijo el k busco el alpha
``` python
accs = []
alphas = list(range(1, 101, 3))
for alpha in alphas:
    # Ya V la tengo calculada
    X_pca_train = X_train @ V[:, :alpha]
    X_pca_test = X_test @ V[:, :alpha]
    
    ## Creo y entreno
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_pca_train, y_train)

    # Predigo
    y_pred = clf.predict(X_pca_test)

    # Me fijo el accuracy
    acc = accuracy_score(y_test, y_pred)
    print("{:<2} ----> {:.3f}".format(alpha, acc))
    accs.append(acc)
```

Buscando k y alpha a la vez
Ahora, elegimos primero el k, luego el α...pero ¿y si había algún otro k que me de una mejor performance optimizando α?

Solución: buscar conjuntamente k y α....aunque esto puede ser costoso!
``` python
pruebas = []

alphas = list(range(1, 91, 5))

for k in [1, 3, 5, 7, 9, 11, 13]:
    alphas = list(range(5, 91, 5))
    
    for alpha in alphas:
        X_pca_train = X_train @ V[:, :alpha]
        X_pca_test = X_test @ V[:, :alpha]

        ## Creo y entreno
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_pca_train, y_train)

        # Predigo
        y_pred = clf.predict(X_pca_test)

        # Me fijo el accuracy
        acc = accuracy_score(y_test, y_pred)
        print("k = {:<2} alpha = {:<2} ----> {:.3f}".format(k, alpha, acc))
        
        pruebas.append({
            "k": k,
            "alpha": alpha,
            "acc": acc,
        })
    
```


