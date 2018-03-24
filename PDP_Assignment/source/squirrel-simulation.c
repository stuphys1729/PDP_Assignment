/*
*	Author: B059476
*	This program models red squirrels
*/

#include <stdio.h>
#include "mpi.h"
#include "pool.h"

#define init_squirrels 34
#define num_env_cells 16

static void workerCode();
static void squirrelCode();
static void environmentCode();

MPI_Comm comw = MPI_COMM_WORLD;

int main(int argc, char* argv[]) {
	// Call MPI initialize first
	MPI_Init(&argc, &argv);

	//Initialise the process pool.
	int statusCode = processPoolInit();

	if (statusCode == 1) {
		// A worker so do the worker tasks
		workerCode();
	}
	else if (statusCode == 2) {
		/*
		* This is the master, each call to master poll will block until a message is received and then will handle it and return
		* 1 to continue polling and running the pool or 0 to quit.
		*/
		int i, active_squirrels = 0, infected_squirrels = 0, returnCode;

		// Initialise the squirrels
		MPI_Request initial_squirrel_requests[init_squirrels];
		for (i = 0; i<init_squirrels; i++) {
			int workerPid = startWorkerProcess();
			MPI_Irecv(NULL, 0, MPI_INT, workerPid, 0, comw, &initial_squirrel_requests[i]);
			active_squirrels++;
			printf("Master started squirrel %d on MPI process %d\n", i, workerPid);
		}

		// Initialise the environment cells
		MPI_Request environment_requests[num_env_cells];
		for (i = 0; i < num_env_cells; i++) {
			int workerPid = startWorkerProcess();
			MPI_Irecv(NULL, 0, MPI_INT, workerPid, 0, comw, &environment_requests[i]);
			printf("Master started squirrel %d on MPI process %d\n", i, workerPid);
		}


		// Run the simulation
		int masterStatus = masterPoll();
		while (masterStatus) {
			masterStatus = masterPoll();
			for (i = 0; i<10; i++) {
				// Checks all outstanding workers that master spawned to see if they have completed
				if (initial_squirrel_requests[i] != MPI_REQUEST_NULL) {
					MPI_Test(&initial_squirrel_requests[i], &returnCode, MPI_STATUS_IGNORE);
					if (returnCode) active_squirrels--;
				}
			}
			// If we have no live squirrels, or too many, then the simulation stops.
			if (active_squirrels == 0 || active_squirrels > 199) break;
		}
	}
	// Finalizes the process pool, call this before closing down MPI
	processPoolFinalise();
	// Finalize MPI, ensure you have closed the process pool first
	MPI_Finalize();
	return 0;
}

static void workerCode() {
	int workerStatus = 1;

	while (workerStatus) {
		int function = getCommandData(); // The wake-up data tells us if this process is a squirrel or environment cell
		if (function == 1) {
			// This is an environment cell
			environmentCode();
		}
		else if (function == 2) {
			// This is a squirrel
			squirrelCode();
		}
		workerStatus = workerSleep();	// This MPI process will sleep, further workers may be run on this process now
	}
}

static void squirrelCode()
{
	int workerStatus = 1;

	while (workerStatus) {
		// Do squirrel stuff

		workerStatus = shouldWorkerStop(); // If the simulation has been ended, this worker should stop
	}
}

static void environmentCode() {
	int workerStatus = 1;
	
	while (workerStatus) {
		// Do environment stuff

		workerStatus = shouldWorkerStop(); // If the simulation has been ended, this worker should stop
	}
}
	
