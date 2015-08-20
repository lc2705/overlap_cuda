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
	ACSM_STATETABLE *table = (ACSM_STATETABLE*)AC_MALLOC(sizeof(ACSM_STATETABLE) * acsm->acsmMaxStates);
	MEMASSERT(table,"StateTable Create");
	memset(table,0,acsm->acsmMaxStates);
	
	//initialize all States NextStates to ACSM_FAILED_STATE 
	for(i = 0 ; i < acsm->acsmMaxStates ; i++)
	{
		for(j = 0; j < CH_SIZE; j++)
		{
			table[i].NextState[j] = ACSM_FAILED_STATE;
		}
	}
	
	for(pat = acsm->acsmPatterns ; pat != NULL ; pat = pat->next)
	{
		i = 0;
		unsigned char *tmp_patrn = pat->patrn;
		table[i]
	}
	
}

//create the state table in device according to the table in host
void acsmCreateDeviceStateTable(ACSM_STRUCT *acsm)
{
	
}

void acsmHostSearch(ACSM_STRUCT *acsm,char *input_file,MATCH_RESULT *result_array)
{
	
}

void acsmDeviceSearch(ACSM_STRUCT *acsm,char *input_file,MATCH_RESULT *result_array)
{
	
}

void acsmResultProcess(ACSM_STRUCT *acsm,MATCH_RESULT *result_array)
{
	
}

void acsmFree(ACSM_STRUCT * acsm)
{
	
}



