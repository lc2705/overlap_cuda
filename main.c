#include "ac.h"

int main(int argc,char **argv)
{
	char *pattern_file = "pattern.txt";
	char *input_file = "data.txt";
	int input_size;
	int *match_result;

	ACSM_STRUCT *acsm = acsmNew(100);
	
	acsmAddPatternFromFile(acsm,pattern_file);

	acsmCreateHostStateTable(acsm);
	acsmCreateDeviceStateTable(acsm);

	acsmHostSearchFromFile(acsm,input_file,&input_size,matched_result);

	acsmResultProcess(acsm,matched_result,input_size);

	acsmFree(acsm);

	return 0;
}
