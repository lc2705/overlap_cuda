

#ifndef AC_H
#define AC_H

#define CH_SIZE 256
#define MAX_PATTERN_LEN 200
#define ACSM_FAILED_STATE -1

typedef struct _acsm_pattern
{
	struct _acsm_pattern	*next;  
	unsigned char 			*patrn; //point to the pattern buffer
	int 					n;  	//length of this pattern 
	int 					*nmatch_array;
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
	
	int MatchandDepth;  /*  0xF0000000 && MatchandDepth == 0 -> match  
						 *  0xF0000000 && MatchandDepth == 0xF0000000  ->  no match 	
						 *  0x0FFFFFFF && MatchandDepth == Depth in ACSM_STATETABLE 
						 */
}ACSM_STATETABLE_CUDA;

typedef struct
{
	ACSM_STATETABLE 		*acsmStateTable;
	ACSM_STATETABLE_CUDA	*acsmStateTableCuda;
	
	ACSM_PATTERN 			*acsmPatterns;
	
	int acsmMaxStates;
	int acsmNumstates;
	int acsmNumThreads;
}ACSM_STRUCT;

typedef struct
{
	int Position;  //the position where match happened
	int State;     
}MATCH_RESULT;

//initialize acsm struct with the num of threads
ACSM_STRUCT * acsmNew (int numThread); 

void acsmAddPatternFromFile(ACSM_STRUCT *acsm, char *filename);

//create state table and compile it in host
void acsmCreateHostStateTable(ACSM_STRUCT *acsm);    

//create the state table in device according to the table in host
void acsmCreateDeviceStateTable(ACSM_STRUCT *acsm);  

void acsmHostSearch(ACSM_STRUCT *acsm,char *input_file,MATCH_RESULT *result_array);

void acsmDeviceSearch(ACSM_STRUCT *acsm,char *input_file,MATCH_RESULT *result_array);

void acsmResultProcess(ACSM_STRUCT *acsm,MATCH_RESULT *result_array);

void acsmFree(ACSM_STRUCT * acsm);


#endif
