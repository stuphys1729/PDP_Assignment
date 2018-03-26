/*
*	Author: B059476
*	This program models red squirrels
*/

#include <stdio.h>
#include "mpi.h"
#include "pool.h"

#include "squirrel-functions.h"

#define init_squirrels 34
#define num_env_cells 16
#define equil_steps 20

#define SQUIRREL_STEP 123
#define AVG_POP 124
#define AVG_INF 125

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
		int i, active_squirrels = 0, infected_squirrels = 0, returnCode, env_cell_ids[num_env_cells];

		// Initialise the environment cells
		MPI_Request environment_requests[num_env_cells];
		for (i = 0; i < num_env_cells; i++) {
			int workerPid = startWorkerProcess();
			env_cell_ids[i] = workerPid;
			MPI_Irecv(NULL, 0, MPI_INT, workerPid, 0, comw, &environment_requests[i]);
			printf("Master started squirrel %d on MPI process %d\n", i, workerPid);
		}

		// Initialise the squirrels
		MPI_Request initial_squirrel_requests[init_squirrels];
		for (i = 0; i<init_squirrels; i++) {
			int workerPid = startWorkerProcess();
			MPI_Irecv(NULL, 0, MPI_INT, workerPid, 0, comw, &initial_squirrel_requests[i]);
			active_squirrels++;
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
	struct PP_Data data = getCommandData(); // The wake-up data tells us if this process is a squirrel or environment cell

	while (workerStatus) {
		
		if (data.function == 1) {
			// This is an environment cell
			environmentCode();
		}
		else if (data.function == 2) {
			// This is a squirrel
			squirrelCode(data.cells);
		}
		workerStatus = workerSleep();	// This MPI process will sleep, further workers may be run on this process now
	}
}

static void squirrelCode(int* data)
{
	int alive = 1, infected = 0, my_rank, i, cell, cell_proc, steps_since_inf;
	int not_send = 0, inf_send = 1; // Different buffers so we don't overwrite when squirrel becomes infected
	float x=0, y=0, x_new, y_new, avg_pop, avg_inf;
	long state = -1 - my_rank;
	MPI_Request step_send, pop_recv, inf_recv;

	MPI_Comm_rank(comw, &my_rank);

	initialiseRNG(&state); // Initialise random number generation

	// Let the squirrel get to a position independent of the start
	for (i = 0; i < equil_steps; i++) {
		squirrelStep(x, y, &x_new, &y_new, &state);
		x = x_new; y = y_new;
	}
	

	while (alive) {
		// Do squirrel stuff
		// Step to new position
		squirrelStep(x, y, &x_new, &y_new, &state);
		x = x_new; y = y_new;

		// Find out the corresponding environment cell
		cell = getCellFromPosition(x, y);
		// And the corresponding governing process
		cell_proc = data[cell];
		// Notify the cell that we have stepped in to it, and whether we are infected
		if (infected) {
			MPI_Ssend(&inf_send, 1, MPI_INT, cell_proc, SQUIRREL_STEP, comw, &step_send);
			steps_since_inf++;
			if (steps_since_inf > 50) alive = willDie(&state);
		}
		else {
			MPI_Ssend(&not_send, 1, MPI_INT, cell_proc, SQUIRREL_STEP, comw, &step_send);
		}

		// Receive the corresponding population and infection levels
		MPI_Recv(&avg_pop, 1, MPI_FLOAT, cell_proc, AVG_POP, comw, &pop_recv);
		MPI_Recv(&avg_inf, 1, MPI_FLOAT, cell_proc, AVG_INF, comw, &inf_recv);

		if (willGiveBirth(avg_pop, &state)) {

		}

		if (!infected) infected = willCatchDisease(avg_inf, &state);

		if (shouldWorkerStop()) { break; } // If the simulation has been ended, this worker should stop
	}
}

static void environmentCode() {
	int workerStatus = 1;
	
	while (workerStatus) {
		// Do environment stuff

		workerStatus = shouldWorkerStop(); // If the simulation has been ended, this worker should stop
	}
}
	
