
#ifndef GET_INITIAL_H
#define GET_INITIAL_H

#ifdef fortran1
#define f_get_initial_kerrschild get_initial_kerrschild
#define f_get_initial_kerrschild_ss get_initial_kerrschild_ss
#define f_get_initial_single get_initial_bssn3
#define f_get_ansorg_single get_ansorg_single
#define f_get_initial_binary get_initial_bssn6
#define f_get_ansorg_binary get_ansorg_binary
#define f_get_ansorg_nbhs get_ansorg_nbhs
#define f_get_ansorg_nbhs_ss get_ansorg_nbhs_ss
#define f_get_initial_postdeal get_initial_postdeal
#define f_get_initial_nbhs get_initial_nbhs
#define f_get_lousto_nbhs get_lousto_nbhs
#define f_get_pablo_nbhs get_pablo_nbhs
#define f_get_shapiro get_shapiro
#define f_get_niall_minkowski get_niall_minkowski
#endif
#ifdef fortran2
#define f_get_initial_kerrschild GET_INITIAL_KERRSCHILD
#define f_get_initial_kerrschild_ss GET_INITIAL_KERRSCHILD_SS
#define f_get_initial_single GET_INITIAL_BSSN3
#define f_get_ansorg_single GET_ANSORG_SINGLE
#define f_get_initial_binary GET_INITIAL_BSSN6
#define f_get_ansorg_binary GET_ANSORG_BINARY
#define f_get_ansorg_nbhs GET_ANSORG_NBHS
#define f_get_ansorg_nbhs_ss GET_ANSORG_NBHS_SS
#define f_get_initial_postdeal GET_INITIAL_POSTDEAL
#define f_get_initial_nbhs GET_INITIAL_NBHS
#define f_get_lousto_nbhs GET_LOUSTO_NBHS
#define f_get_pablo_nbhs GET_PABLO_NBHS
#define f_get_shapiro GET_SHAPIRO
#define f_get_niall_minkowski GRT_NIALL_MINKOWSKI
#endif
#ifdef fortran3
#define f_get_initial_kerrschild get_initial_kerrschild_
#define f_get_initial_kerrschild_ss get_initial_kerrschild_ss_
#define f_get_initial_single get_initial_bssn3_
#define f_get_ansorg_single get_ansorg_single_
#define f_get_initial_binary get_initial_bssn6_
#define f_get_ansorg_binary get_ansorg_binary_
#define f_get_ansorg_nbhs get_ansorg_nbhs_
#define f_get_ansorg_nbhs_ss get_ansorg_nbhs_ss_
#define f_get_initial_postdeal get_initial_postdeal_
#define f_get_initial_nbhs get_initial_nbhs_
#define f_get_lousto_nbhs get_lousto_nbhs_
#define f_get_pablo_nbhs get_pablo_nbhs_
#define f_get_shapiro get_shapiro_
#define f_get_niall_minkowski get_niall_minkowski_
#endif

extern "C"
{
    void f_get_initial_kerrschild(int *, double *, double *, double *,
                                  double *, double *,
                                  double *, double *, double *, double *, double *, double *,
                                  double *, double *, double *, double *, double *, double *,
                                  double *, double *, double *,
                                  double *, double *, double *, double *,
                                  double *, double *, double *);
}

extern "C"
{
    void f_get_initial_kerrschild_ss(int *, double *, double *, double *,
                                     double *, double *,
                                     double *, double *, double *, double *, double *, double *,
                                     double *, double *, double *, double *, double *, double *,
                                     double *, double *, double *,
                                     double *, double *, double *, double *,
                                     double *, double *, double *);
}

extern "C"
{
    void f_get_initial_single(int *, double *, double *, double *,
                              double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *,
                              double *, double *, double *, double *,
                              double *, double *, double *,
                              double &, double *, double *, double *);
}

extern "C"
{
    void f_get_initial_binary(int *, double *, double *, double *,
                              double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *,
                              double *, double *, double *, double *,
                              double *, double *, double *,
                              double *, double *, double *, double *);
}

extern "C"
{
    void f_get_ansorg_single(int *, double *, double *, double *,
                             double *, double *,
                             double *, double *, double *, double *, double *, double *,
                             double *, double *, double *, double *, double *, double *,
                             double *, double *, double *,
                             double *, double *, double *, double *,
                             double *, double *, double *,
                             double &, double *, double *, double *);
}

extern "C"
{
    void f_get_ansorg_binary(int *, double *, double *, double *,
                             double *, double *,
                             double *, double *, double *, double *, double *, double *,
                             double *, double *, double *, double *, double *, double *,
                             double *, double *, double *,
                             double *, double *, double *, double *,
                             double *, double *, double *,
                             double *, double *, double *, double *);
}

extern "C"
{
    void f_get_ansorg_nbhs(int *, double *, double *, double *,
                           double *, double *,
                           double *, double *, double *, double *, double *, double *,
                           double *, double *, double *, double *, double *, double *,
                           double *, double *, double *,
                           double *, double *, double *, double *,
                           double *, double *, double *,
                           double *, double *, double *, double *, int &);
}

extern "C"
{
    void f_get_ansorg_nbhs_ss(int *, double *, double *, double *,
                              double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *, double *, double *, double *,
                              double *, double *, double *,
                              double *, double *, double *, double *,
                              double *, double *, double *,
                              double *, double *, double *, double *, int &);
}

extern "C"
{
    void f_get_initial_postdeal(int *, double *, double *, double *,
                                double *, double *,
                                double *, double *, double *, double *, double *, double *,
                                double *, double *, double *, double *, double *, double *,
                                double *, double *, double *,
                                double *, double *, double *, double *,
                                double *, double *, double *);
}

extern "C"
{
    void f_get_lousto_nbhs(int *, double *, double *, double *,
                           double *, double *,
                           double *, double *, double *, double *, double *, double *,
                           double *, double *, double *, double *, double *, double *,
                           double *, double *, double *,
                           double *, double *, double *, double *,
                           double *, double *, double *,
                           double *, double *, double *, double *, int &);
}

extern "C"
{
    void f_get_initial_nbhs(int *, double *, double *, double *,
                            double *, double *,
                            double *, double *, double *, double *, double *, double *,
                            double *, double *, double *, double *, double *, double *,
                            double *, double *, double *,
                            double *, double *, double *, double *,
                            double *, double *, double *,
                            double *, double *, double *, double *, int &);
}

extern "C"
{
    void f_get_pablo_nbhs(int *, double *, double *, double *,
                          double *, double *,
                          double *, double *, double *, double *, double *, double *,
                          double *, double *, double *, double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *, double *, int &);
}

extern "C"
{
    void f_get_shapiro(int *, double *, double *, double *,
                       double *, double *,
                       double *, double *, double *, double *, double *, double *,
                       double *, double *, double *, double *, double *, double *,
                       double *, double *, double *,
                       double *, double *, double *, double *,
                       double *, double *, double *, int &);
}

extern "C"
{
    void f_get_niall_minkowski(int *, double *, double *, double *,
                               double *, double *,
                               double *, double *, double *, double *, double *, double *,
                               double *, double *, double *, double *, double *, double *,
                               double *, double *, double *,
                               double *, double *, double *, double *,
                               double *, double *, double *);
}

#endif /* GET_INITIAL_H */
