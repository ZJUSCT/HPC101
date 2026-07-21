
#ifndef VAR_H
#define VAR_H

class var
{

public:
   char name[20];
   int sgfn;
   double SoA[3];
   double propspeed;

public:
   var(const char *namei, int sgfni,
       const double SYM1, const double SYM2, const double SYM3);
   // original interface:
   // var(char *namei, int sgfni,
   //     const double SYM1, const double SYM2, const double SYM3);

   ~var();

   void setpropspeed(const double vl);
};

#endif /* VAR_H */
