#include <stdbool.h>

typedef struct ngcomplex {
    double cx_real;
    double cx_imag;
} ngcomplex_t;

typedef bool NG_BOOL;

/* vector info obtained from any vector in ngspice.dll.
   Allows direct access to the ngspice internal vector structure,
   as defined in include/ngspice/devc.h . */
typedef struct vector_info {
    char *v_name;		/* Same as so_vname. */
    int v_type;			/* Same as so_vtype. */
    short v_flags;		/* Flags (a combination of VF_*). */
    double *v_realdata;		/* Real data. */
    ngcomplex_t *v_compdata;	/* Complex data. */
    int v_length;		/* Length of the vector. */
} vector_info, *pvector_info;

typedef struct vecvalues {
    char* name;        /* name of a specific vector */
    double creal;      /* actual data value */
    double cimag;      /* actual data value */
    NG_BOOL is_scale;     /* if 'name' is the scale vector */
    NG_BOOL is_complex;   /* if the data are complex numbers */
} vecvalues, *pvecvalues;

typedef struct vecvaluesall {
    int veccount;      /* number of vectors in plot */
    int vecindex;      /* index of actual set of vectors. i.e. the number of accepted data points */
    pvecvalues *vecsa; /* values of actual set of vectors, indexed from 0 to veccount - 1 */
} vecvaluesall, *pvecvaluesall;

/* info for a specific vector */
typedef struct vecinfo
{
    int number;     /* number of vector, as postion in the linked list of vectors, starts with 0 */
    char *vecname;  /* name of the actual vector */
    NG_BOOL is_real;   /* TRUE if the actual vector has real data */
    void *pdvec;    /* a void pointer to struct dvec *d, the actual vector */
    void *pdvecscale; /* a void pointer to struct dvec *ds, the scale vector */
} vecinfo, *pvecinfo;

/* info for the current plot */
typedef struct vecinfoall
{
    /* the plot */
    char *name;
    char *title;
    char *date;
    char *type;
    int veccount;

    /* the data as an array of vecinfo with length equal to the number of vectors in the plot */
    pvecinfo *vecs;

} vecinfoall, *pvecinfoall;

/* callback functions
addresses received from caller with ngSpice_Init() function
*/
/* sending output from stdout, stderr to caller */
typedef int (SendChar)(char*, int, void*);
/*
   char* string to be sent to caller output
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller, e.g. pointer to object having sent the request
*/
/* sending simulation status to caller */
typedef int (SendStat)(char*, int, void*);
/*
   char* simulation status and value (in percent) to be sent to caller
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller
*/
/* asking for controlled exit */
typedef int (ControlledExit)(int, NG_BOOL, NG_BOOL, int, void*);
/*
   int   exit status
   NG_BOOL  if true: immediate unloading dll, if false: just set flag, unload is done when function has returned
   NG_BOOL  if true: exit upon 'quit', if false: exit due to ngspice.dll error
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller
*/
/* send back actual vector data */
typedef int (SendData)(pvecvaluesall, int, int, void*);
/*
   vecvaluesall* pointer to array of structs containing actual values from all vectors
   int           number of structs (one per vector)
   int           identification number of calling ngspice shared lib
   void*         return pointer received from caller
*/

/* send back initailization vector data */
typedef int (SendInitData)(pvecinfoall, int, void*);
/*
   vecinfoall* pointer to array of structs containing data from all vectors right after initialization
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* indicate if background thread is running */
typedef int (BGThreadRunning)(NG_BOOL, int, void*);
/*
   NG_BOOL        true if background thread is running
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* callback functions
   addresses received from caller with ngSpice_Init_Sync() function
*/

/* ask for VSRC EXTERNAL value */
typedef int (GetVSRCData)(double*, double, char*, int, void*);
/*
   double*     return voltage value
   double      actual time
   char*       node name
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* ask for ISRC EXTERNAL value */
typedef int (GetISRCData)(double*, double, char*, int, void*);
/*
   double*     return current value
   double      actual time
   char*       node name
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* ask for new delta time depending on synchronization requirements */
typedef int (GetSyncData)(double, double*, double, int, int, int, void*);
/*
   double      actual time (ckt->CKTtime)
   double*     delta time (ckt->CKTdelta)
   double      old delta time (olddelta)
   int         redostep (as set by ngspice)
   int         identification number of calling ngspice shared lib
   int         location of call for synchronization in dctran.c
   void*       return pointer received from caller
*/

int  ngSpice_Init(SendChar* printfcn, SendStat* statfcn, ControlledExit* ngexit,
                  SendData* sdata, SendInitData* sinitdata, BGThreadRunning* bgtrun, void* userData);
int  ngSpice_Init_Sync(GetVSRCData *vsrcdat, GetISRCData *isrcdat, GetSyncData *syncdat, int *ident, void *userData);
int  ngSpice_Command(char* command);
pvector_info ngGet_Vec_Info(char* vecname);

int ngSpice_Circ(char** circarray);
char* ngSpice_CurPlot(void);
char** ngSpice_AllPlots(void);
char** ngSpice_AllVecs(char* plotname);
NG_BOOL ngSpice_running(void);
NG_BOOL ngSpice_SetBkpt(double time);
