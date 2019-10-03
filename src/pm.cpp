Vector PCA::powerMethod(MatrixXd A, Vector x)
{
    Vector act = x;
    for(int k = 0; k< x.rows(); k++) //Criterio de parada?
    {
        Vector Ax_t = A * act;
        act = Ax_t / Ax_t.norm();
    }
    return act;
}
