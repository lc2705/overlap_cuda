

#ifndef AC_H
#define AC_H

#define CH_SIZE 256
#define MAX_PATTERN_LEN 200
#define MAX_LINE 100
#define MAX_STRING_LEN 1500
#define ACSM_FAILED_STATE -1

typedef struct _acsm_pattern
{
	struct _acsm_pattern	*next;  
	unsigned char 			*patrn; //point to the pattern buffer
	int 					n;  	//length of this pattern 
}ACSM_PATTERN;

typedef struct
{
	int NextState[CH_SIZE];
	
	int FailState;     //only used for transform NFA to DFA
	
	ACSM_PATTERN *MatchList;   //Matched Pattern at this position
	
	int Depth;    // depth in the tree , used for DCPM 
	 
}ACSM_STATETABLE;

typedef struct
{
	int NextState[CH_SIZE];
	
	int Depth;  

	char Accept; // 1:this is an accept state  
				 // 0: not accepted

}ACSM_STATETABLE_CUDA;

typedef struct
{
	ACSM_STATETABLE 		*acsmStateTable;
	ACSM_STATETABLE_CUDA	*acsmStateTableCuda_h;
	ACSM_STATETABLE_CUDA	*acsmStateTableCuda_d;
	
	ACSM_PATTERN 			*acsmPatterns;
	
	int acsmMaxStates;
	int acsmNumStates;
	int acsmNumThreads;

}ACSM_STRUCT;


//initialize acsm struct with the num of threads
ACSM_STRUCT * acsmNew (int numThread); 

void acsmAddPatternFromFile(ACSM_STRUCT *acsm, char *filename);

//create state table and compile it in host
void acsmCreateHostStateTable(ACSM_STRUCT *acsm);    

//create the state table in device according to the table in host
void acsmCreateDeviceStateTable(ACSM_STRUCT *acsm);  

void acsmHostSearchFromFile(ACSM_STRUCT *acsm,char *input_file,int *input_size,int *matched_result);

void acsmResultProcess(ACSM_STRUCT *acsm,int *matched_result,int input_size);

void acsmFree(ACSM_STRUCT * acsm);


#endif
