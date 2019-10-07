#include "pca.h"

void calcularPca(vector<entrada> &etiquetados, vector<entrada> &sinEtiquetar, fstream &myfile, int cantidadAutovectores,Matriz* X_t) {
    // calculamos la matriz de covarianza
    Matriz *covarianza = matrizDeCovarianza(etiquetados,X_t);
    std::vector<vectorNum*> autovectores;

    myfile.precision(6);
    myfile << scientific;
    //Empiezo a conseguir autovectores
    for (int i = 0; i < cantidadAutovectores; i++) {
        vectorNum * autovector = metodoDeLasPotencias(covarianza);
        //guardo los actovectores en un vector
        autovectores.push_back(autovector);
        double lamda = encontrarAutovalor(autovector, covarianza);

        myfile << lamda << endl;

        vectorNum *autovectorAux = autovector->copy();
        autovector->multiplicacionEscalar(lamda);
        covarianza->restarMatriz(multiplicacionVectTrans2(autovector,autovectorAux));
        delete autovectorAux;
    }
    trasponerEntrada(etiquetados, autovectores, cantidadAutovectores);
    trasponerEntrada(sinEtiquetar, autovectores, cantidadAutovectores);

    delete covarianza;
    for (int i = 0; i < autovectores.size(); i++) {
        delete autovectores[i];
    }
}

Matriz *matrizDeCovarianza(vector<entrada> &etiquetados,Matriz *X_t) {
    Matriz *matrizCovarianza = matCovarianza(etiquetados, X_t);
    return matrizCovarianza;
}
