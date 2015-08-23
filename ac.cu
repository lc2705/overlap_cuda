#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ac.h"

//=============Assistant Function==================

#define MEMASSERT(p,s) if(!p){\
							fprintf(stderr,"ACSM-No Memory: %s!\n",s);\
							exit(0);\
						}

static void *AC_MALLOC (int n) 
{
    void *p;
    p = malloc (n);

    return p;
}


static void AC_FREE (void *p) 
{
    if (p)
        free (p);
}

/*
*    Simple QUEUE NODE
*/ 
typedef struct _qnode
{
    int state;
    struct _qnode *next;
}QNODE;

/*
*    Simple QUEUE Structure
*/ 
typedef struct _queue
{
    QNODE * head, *tail;
    int count;
}QUEUE;

/*
*Init the Queue
*/ 
static void queue_init (QUEUE * s) 
{
    s->head = s->tail = NULL;
    s->count = 0;
}


/*
*  Add Tail Item to queue
*/ 
static void queue_add (QUEUE * s, int state) 
{
    QNODE * q;
    /*Queue is empty*/
    if (!s->head)
    {
        q = s->tail = s->head = (QNODE *) AC_MALLOC (sizeof (QNODE));
        
		/*if malloc failed,exit the problom*/
        MEMASSERT (q, "queue_add");
       
	    q->state = state;
        q->next = 0; /*Set the New Node's Next Null*/
    }
    else
    {
        q = (QNODE *)AC_MALLOC (sizeof (QNODE));
        
		MEMASSERT (q, "queue_add");
        q->state = state;
        q->next = 0;
        
		/*Add the new Node into the queue*/
        s->tail->next = q;
        
		/*set the new node is the Queue's Tail*/
        s->tail = q;
    }
    s->count++;
}


/*
*  Remove Head Item from queue
*/ 
static int queue_remove (QUEUE * s) 
{
    int state = 0;
    QNODE * q;
    /*Remove A QueueNode From the head of the Queue*/
    if (s->head)
    {
        q = s->head;
        state = q->state;
        s->head = s->head->next;
        s->count--;

        /*If Queue is Empty,After Remove A QueueNode*/
        if (!s->head)
        {
            s->tail = 0;
            s->count = 0;
        }
        /*Free the QueNode Memory*/
        AC_FREE (q);
    }
    return state;
}


/*
*Return The count of the Node in the Queue
*/ 
static int queue_count (QUEUE * s) 
{
    return s->count;
}


/*
*Free the Queue Memory
*/ 
static void queue_free (QUEUE * s) 
{
    while (queue_count (s))
    {
        queue_remove (s);
    }
}

//===============Assistant Function=============

//add pattern states according to patterns recorded in acsmPatterns, 
//called by acsmCreateHostStateTable
void AddPatternStates(ACSM_STRUCT *acsm,ACSM_PATTERN *p)
{
    unsigned char * pattern = p->patrn;
    int nextstate, state = 0;
    int n = p->n;

    //match up pattern with existing states
    for( ; n > 0 ; n--)
    {
        nextstate = acsm->acsmStateTable[state].NextState[*pattern];
        if(nextstate == ACSM_FAILED_STATE)
            break;
        state = nextstate;
        pattern++;
    }	
    
    //add new states for the rest of the pattern bytes
    for( ; n > 0 ; n--)
    {
		acsm->acsmNumStates++;
		acsm->acsmStateTable[state].NextState[*pattern] = acsm->acsmNumStates;
		acsm->acsmStateTable[acsm->acsmNumStates].Depth = acsm->acsmStateTable[state].Depth + 1;
		state = acsm->acsmNumStates;
		pattern++;
    }

	//arrive at an accept state, so add into the Matchlist of the state
	ACSM_PATTERN *newp = (ACSM_PATTERN*) AC_MALLOC (sizeof(ACSM_PATTERN));
	MEMASSERT(newp,"add match list");
	memcpy(newp,p,sizeof(ACSM_PATTERN));

	newp->next = acsm->acsmStateTable[state].MatchList;
	acsm->acsmStateTable[state].MatchList = newp;
}

//build NFA,called by acsmCreateHostStateTable
static void Build_NFA(ACSM_STRUCT *acsm)
{
	int i;
	int state;
	QUEUE state_q;
	queue_init(&state_q);

	for(i = 0 ; i < CH_SIZE ; i++)
	{
		state = acsm->acsmStateTable[0].NextState[i];
		if(state)
		{
			queue_add(&state_q,state);
			acsm->acsmStateTable[state].FailState = 0;
		}
	}

	int pre_s,next_s,fail_s;
	ACSM_PATTERN * m_list;
	while(queue_count (&state_q) > 0)
	{
		pre_s = queue_remove(&state_q);

		for(i = 0 ; i < CH_SIZE ; i++)
		{
			s = acsm->acsmStateTable[pre_s].NextState[i];
			
			//only deal with the valid states
			if(s != ACSM_FAILED_STATE)
			{
				queue_add(&state_q, s);
				fail_s = acsm->acsmStateTable[pre_s].FailState;

				next_s = acsm->acsmStateTable[fail_s].NextState[i];
				while(next_s == ACSM_FAIL_STATE)
				{
					fail_s = acsm->acsmStateTable[fail_s].FailState;
					next_s = acsm->acsmStateTable[fail_s].NextState[i];
				}

				acsm->acsmStateTable[s].FailState = next_s;
				
				//if state s 's failstate is an accept state,failstate's match list should be insert into s 's
				ACSM_PATTERN * newp;
				for(m_list = acsm->acsmStateTable[next_s].MatchList;
					m_list != NULL ;
					m_list = m_list->next )
				{
					newp = (ACSM_PATTERN*) AC_MALLOC (sizeof(ACSM_PATTERN));
					MEMASSERT(newp,"match list copying during NFA building");
					memcpy(mewp,m_list,sizeof(ACSM_PATTERN));

					newp->next = acsm->acsmStateTable[s].MatchList;
					acsm->acsmStateTable[s].MatchList = newp;
				}
			}
			
		}
	}

	queue_free(&state_q);
}

//build DFA from NFA, called by acsmCreateHostStateTable
static void Convert_NFA_To_DFA(ACSM_STRUCT *acsm)
{
	int i;
	int pre_s,s;
	QUEUE state_q;
	queue_init(&state_q);

	for(i = 0 ; i < CH_SIZE ; i++)
	{
		s = acsm->acsmStateTable[0].NextState[i];
		if(s)
		{
			queue_add(&state_q,s);
		}
	}

	while(queue_count(&state_q) > 0)
	{
		pre_s = queue_remove(&state_q);

		for(i = 0 ; i <	CH_SIZE ; i++)
		{
			s = acsm->acsmStateTable[pre_s].NextState[i];
			if(s != ACSM_FAILED_STATE)
			{
				queue_add(queue,s);
			}
			else
			{
				acsm->acsmStateTable[pre_s].NextState[i] = 
					acsm->acsmStateTable[acsm->acsmStateTable[pre_s].FailState].NextState[i];
			}

		}
	}

	queue_free(&state_q);
}
 
 //=================CUDA_kERNEL=================
 
__global__ void kernelSearch(ACSM_STRUCT *acsm,char *input_string,int *matched_result,int input_size)
{
	int state = 0;
	ACSM_STATETABLE_CUDA *table = acsm->acsmStateTableCuda_d;
	
	int frag_id = threadIdx.x;
	int frag_size = input_size/acsm->acsmNumThreads;
	int frag_beg = frag_id * frag_size;
	int frag_end = frag_beg + frag_size;
	if (frag_id == threadIdx.x - 1)
		frag_end = input_size;

	//stage 1
	for(i = frag_beg; i < frag_end ; i++)
	{
		state = table[state].NextState[input_string[i]];
		if(table[state].Accept)
		{
			matched_result[i] = state;
		}
	}

	//stage 2
	int step = 0;
	for( ; i < input_size ; i++)
	{
		if(table[state].Depth <= step)
			break;
		
		state = table[state].NextState[ input_string[i] ];
		if(table[state].Accept)
			matched_result[i] = state;
		
		step++;
	}
	
}


 //==================API========================

/*
* Init the acsm DataStruct
*/ 
ACSM_STRUCT * acsmNew (int numThread) 
{
    ACSM_STRUCT * p;
    
    p = (ACSM_STRUCT *) AC_MALLOC (sizeof (ACSM_STRUCT));
    MEMASSERT (p, "acsmNew");
    
    if (p)
        memset (p, 0, sizeof (ACSM_STRUCT));
    
    p->acsmNumThread = numThread;
    
    return p;
}

void acsmAddPatternFromFile(ACSM_STRUCT *acsm, char *filename)
{
	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"PatternFile Open Error!\n");
		exit(1);
	}
	
	char buffer[MAX_PATTERN_LEN];
	ACSM_PATTERN *pat = NULL;
	
	while(fgets(buffer,MAX_PATTERN_LEN,fp))
	{
		pat = (ACSM_PATTERN*)AC_MALLOC(sizeof(ACSM_PATTERN));
		MEMASSERT(pat,"ADD PATTERN\n");
		
		pat->n = strlen(buffer);
		pat->patrn = (unsigned char*)AC_MALLOC(sizeof(unsigned char)*(pat->n + 1));
		memcpy(pat->patrn,buffer,pat->n);
		memset(pat->patrn + pat->n,0,1);
	
		pat->next = acsm->acsmPatterns;
		acsm->acsmPatterns = pat;
	}
	
	if(!feof(fp))
	{
		fprintf(stderr,"file gets error!\n");
		exit(1);
	}
	
	fclose(fp);
	
	return;
}

//create state table and compile it in host
void acsmCreateHostStateTable(ACSM_STRUCT *acsm)
{
	int i,j;
	
	//malloc and init the StateTable in host
	ACSM_PATTERN *pat = acsm->acsmPatterns;
	while(pat != NULL)
	{
		acsm->acsmMaxStates += pat->n;
		pat = pat->next;
	}
	acsm->acsmStateTable = (ACSM_STATETABLE*)AC_MALLOC(sizeof(ACSM_STATETABLE) * acsm->acsmMaxStates);
	MEMASSERT(acsm->acsmStateTable,"StateTable Create");
	memset(acsm->acsmStateTable,0,acsm->acsmMaxStates);
	
	acsm->acsmNumStates = 0;  //acsm->acsmNumStates - 1 == number of states
	                          //because state 0 is already exited 
	
	//initialize all States NextStates to ACSM_FAILED_STATE 
	for(i = 0 ; i < acsm->acsmMaxStates ; i++)
	{
		for(j = 0; j < CH_SIZE; j++)
		{
			acsm->acsmStateTable[i].NextState[j] = ACSM_FAILED_STATE;
		}
	}
	
	for(pat = acsm->acsmPatterns ; pat != NULL ; pat = pat->next)
	{
		AddPatternStates(acsm,pat);
	}
	
	//build the NFA and then convert it into DFA
	for(i = 0 ; i < CH_SIZE ; i++)
	{
		if(acsm->acsmStateTable[0].NextState[i] == ACSM_FAILED_STATE)
		{
			acsm->acsmStateTable[0].NextState[i] = 0;
		}
	}

	Build_NFA(acsm);

	Convert_NFA_To_DFA (acsm);

}


//create the state table in device according to the table in host
void acsmCreateDeviceStateTable(ACSM_STRUCT *acsm)
{
	int i,j;
	int table_size;
	ACSM_STATETABLE * table;
	ACSM_STATETABLE_CUDA * table_cuda;

	// create StateTableCuda_h
	table_size = sizeof(ACSM_STATEtABLE_CUDA) * (acsm->acsmNumStates + 1);
	acsm->acsmStateTableCuda_h = (ACSM_STATETABLE_CUDA*)AC_MALLOC(table_size);
	MEMASSERT(acsm->acsmStateTableCuda_h,"acsmCreateDeviceStateTable");

	table = acsm->acsmStateTable;
	table_cuda = acsm->acsmStateTableCuda_h;
	for( i = 0 ; i <= acsm->acsmNumStates ; i++)
	{
		for( j = 0 ; j < CH_SIZE ; j++)
		{
			table_cuda[i].NextState[j] = table[i].NextState[j];
		}

		if(table[i].MatchList == NULL)
			table_cuda[i].Accept = 0;
		else
			table_cuda[i].Accept = 1;

		table_cuda[i].Depth = table[i].Depth;
	}

	//create StateTableCuda_d 
	cudaMalloc(&(acsm->acsmStateTableCuda_d),table_size);
	cudaMemcpy(acsm->acsmStateTableCuda_d,acsm->acsmStateTableCuda_h,table_size);

}


void acsmHostSearchFromFile(ACSM_STRUCT *acsm,char *input_file, int *input_size,int *matched_result)
{
	char *input_string;
	int size;

	FILE *fp = fopen(input_file,"rb");
	fseek(fp, 0 , SEEK_END);
	size = ftell (fp);
	rewind(fp);

	input_string = (char *) AC_MALLOC (sizeof(char) * size);
	MEMASSERT(input_string,"acsmHostSearchFromFile");

	matched_result = (int *) AC_MALLOC (sizeof(int) * size);
	MEMASSERT(matched_result,"acsmHostSearchFromFile");
	memset(matched_result,0,sizeof(int) * size);

	size = fread(input_string,1,size,fp);
	fclose(fp);

	cudaError_t error_t;
	char *input_string_d = NULL;
	int *matched_result_d = NULL;

	error_t = cudaMalloc((void **) &input_string_d, sizeof(char) * size);
	if(error_t != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString (error_t) );
		return;
	}
	cudaMemcpy((void *) input_string_d , (void *) input_string , sizeof(char) * size, cudaMemcpyHostToDevice);

	error_t = cudaMalloc((void **) &matched_result_d, sizeof(int) * size);
	if(error_t != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString (error_t) );
		return;
	}
	cudaMemset((void *) matched_result_d , 0 , sizeof(int) * size);

	kernelSearch<<<1,acsm->acsmNumThreads>>>(ACSM_STRUCT *acsm,char *input_string_d,int *matched_result_d,int size);

	cudaMemcpy((void *) matched_result, (void *) matched_result_d, sizeof(int) * size , cudaMemcpyDeviceToHost);

	cudaFree(input_string_d);
	cudaFree(matched_result_d);
	AC_FREE(input_string);
	AC_FREE(matched_result);
	
	*input_size = size;

}

void acsmResultProcess(ACSM_STRUCT *acsm,int *matched_result,int input_size)
{
	int i,state;
	ACSM_PATTERN *mlist;

	for (i = 0; i < input_size ; i++)
	{
		if(matched_result[i])
		{
			state = matched_result;
			mlist = acsm->acsmStateTable[state].MatchList;

			fprintf(stdout,"Position %d\tstate %d\n",i,state);
			for( ; mlist != NULL ; mlist = mlist->next)
			{
				fprintf(stdout,"%s\n",mlist->patrn);
			}
		}
	}
}

void acsmFree(ACSM_STRUCT * acsm)
{
	ACSM_PATTERN *pat,*pat_free;
	
	//free acsmPattern
	for (pat = acsm->acsmPattern ; pat != NULL ; pat = pat->next)
	{
		pat_free = pat;
		AC_FREE(pat_free->patrn);
		AC_FREE(pat_free);
	}

	//free acsmStateTable
	for(int i = 0; i < acsm->acsmMaxStates ; i++)
	{
		for( pat = acsm->acsmStateTable[i].MatchList ; pat != NULL ; pat = pat->next )
		{
			pat_free = pat;
			AC_FREE(pat_free->patrn);
			AC_FREE(pat_free);
		}
	}
	AC_FREE(acsm->acsmStateTable);

	//free acsmStateTableCuda_h
	AC_FREE(acsm->acsmStateTableCuda_h);

	//free acsmStateTableCuda_d
	cudaFree(acsm->acsmStateTableCuda_d);
	
	//free acsm
	AC_FREE(acsm);
}



