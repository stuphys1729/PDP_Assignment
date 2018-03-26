/*
*	Author: B059476
*	This program models red squirrels
*/

#include <stdio.h>
#include "mpi.h"
#include "pool.h"

#include "squirrel-functions.h"

#define init_squirrels 2
#define num_env_cells 16
#define equil_steps 20
#define max_months 2
#define month_time 0.1 // How much real time to use as a simulated month (in seconds)

#define FUNCTION_CALL 112
#define GET_POSITION 113
#define GET_CELLS 114

#define SQUIRREL_STEP 123
#define AVG_POP 124
#define AVG_INF 125

#define DEBUG 1

static double start_time;
static int rank;

static void workerCode();
static void squirrelCode(int);
static void environmentCode(int);
static void debug_msg(char*);
static void error_msg(char*);

MPI_Comm comw = MPI_COMM_WORLD;

int main(int argc, char* argv[]) {
	// Call MPI initialize first
	MPI_Init(&argc, &argv);
	start_time = MPI_Wtime();
	MPI_Comm_rank(comw, &rank);

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
		int f_squirrel = -1; // Defines a squirrel class, a positive integer indicates a certain land cell

		// Initialise the environment cells
		MPI_Request environment_requests[max_months+1][num_env_cells];
		for (i = 0; i < num_env_cells; i++) {
			int workerPid = startWorkerProcess();
			env_cell_ids[i] = workerPid;
			// Tell the processor it's an environmnt cell
			MPI_Isend(&i, 1, MPI_INT, workerPid, FUNCTION_CALL, comw, &environment_requests[0][i]);
			// Check when environment cells complete a month
			MPI_Irecv(NULL, 0, MPI_INT, workerPid, 0, comw, &environment_requests[1][i]);// 1 for month 1
			printf("Master started environment cell %d on MPI process %d\n", i, workerPid);
		}

		// Initialise the squirrels
		MPI_Request initial_squirrel_requests[2][init_squirrels];
		for (i = 0; i<init_squirrels; i++) {
			int workerPid = startWorkerProcess();
			// Tell the processor it's a squirrel
			MPI_Isend(&f_squirrel, 1, MPI_INT, workerPid, FUNCTION_CALL, comw, &initial_squirrel_requests[1][i]);
			// And where the environment cells are
			MPI_Isend(&env_cell_ids, num_env_cells, MPI_INT, workerPid, GET_CELLS, comw, &initial_squirrel_requests[0][i]);
			active_squirrels++;
			printf("Master started squirrel %d on MPI process %d\n", i, workerPid);
		}

		// Run the simulation
		int current_month = 1, month_end = 0;
		MPI_Status environment_statuses[num_env_cells];
		int masterStatus = masterPoll();
		while (masterStatus) {
			masterStatus = masterPoll(); // pass &active_squirrels?
			// Do some stuff

			MPI_Testall(num_env_cells, environment_requests[current_month], &month_end, environment_statuses);
			if (month_end) {
				printf("All cells have completed month %d", current_month);
				printf("Living Squirrels: %d\tInfected Squirrels: %d", active_squirrels, infected_squirrels);
				current_month++;
				if (current_month > max_months) break; // Simulation has ended

				// Preapre messages for next month
				for (i = 0; i < num_env_cells; i++) {
					MPI_Irecv(NULL, 0, MPI_INT, env_cell_ids[i], 0, comw, &environment_requests[current_month][i]);
				}
				month_end = 0;
			}

			// If we have no live squirrels, or too many, then the simulation stops.
			if (active_squirrels > 199) {
				error_msg("Too many Squirrels");
			}
			else if (active_squirrels == 0) {
				error_msg("All the squirrels died :( ");
			}
		}
	}
	// Finalizes the process pool, call this before closing down MPI
	processPoolFinalise();
	// Finalize MPI, ensure you have closed the process pool first
	MPI_Finalize();
	return 0;
}

static void debug_msg(char* msg) {
	double time = start_time - MPI_Wtime();
	printf("[%d] : [Process %d] : %s", time, rank, msg);
}

static void error_msg(char * msg) {
	double time = start_time - MPI_Wtime();
	fprintf(stderr, "[%d] : [Process %d] : %s", time, rank, msg);
	MPI_Abort(comw, 1);
}

static void workerCode() {
	int workerStatus = 1, function;
	MPI_Status function_stat;

	while (workerStatus) {
		int parent = getCommandData(); // The wake-up data tells us who started us
		
		if (parent == 0) { // Master started us, so we could be a land cell
			MPI_Recv(&function, 1, MPI_INT, 0, FUNCTION_CALL, comw, &function_stat);

			if (function > -1) environmentCode(function); // This is a land cell
			else squirrelCode(0); // This is one of the initial squirrels
		}
		else {
			// We were started by another squirrel giving birth to us
			squirrelCode(parent);
		}
		workerStatus = workerSleep();	// This MPI process will sleep, further workers may be run on this process now
	}

}

static void squirrelCode(int parent)
{
	int my_rank, i, cells[num_env_cells];
	float x=0, y=0, x_new, y_new;
	long state = -1 - my_rank;
	MPI_Status cell_recv, pos_recv;

	MPI_Comm_rank(comw, &my_rank);

	initialiseRNG(&state); // Initialise random number generation

	if (parent == 0) {
		// Let the squirrel get to a position independent of the start
		for (i = 0; i < equil_steps; i++) {
			squirrelStep(x, y, &x_new, &y_new, &state);
			x = x_new; y = y_new;
		}
	}
	else {
		MPI_Recv(&x, 1, MPI_FLOAT, parent, GET_POSITION, comw, &pos_recv); // TODO: Make this one message for optimisation
		MPI_Recv(&y, 1, MPI_FLOAT, parent, GET_POSITION, comw, &pos_recv);
	}
	if (DEBUG) {
		char* debug_message;
		sprintf(debug_message, "Squirrel started with pos: (%d,%d)", x, y);
		debug_msg(debug_message);
	}
	// Get the ranks of the environment cells
	MPI_Recv(&cells, num_env_cells, MPI_INT, parent, GET_CELLS, comw, &cell_recv);

	return; // Stop for now (trying to get some meaningful debug output)


	// Simulate the squirrel
	int alive = 1, infected = 0, stepped = 0, cell, cell_proc, steps_since_inf, new_squirrel;
	float avg_pop, avg_inf, x_buf, y_buf;

	while (alive) {
		MPI_Request pos_send, cell_send, step_send;
		MPI_Status pop_recv, inf_recv;
		// Do squirrel stuff
		// Step to new position
		squirrelStep(x, y, &x_new, &y_new, &state);
		x = x_new; y = y_new;

		// Find out the corresponding environment cell
		cell = getCellFromPosition(x, y);
		// And the corresponding governing process
		cell_proc = cells[cell];
		if (DEBUG) {
			char* debug_message;
			sprintf(debug_message, "Squirrel stepped in cell %d on proc %d", cell, cell_proc);
			debug_msg(debug_message);
		}
		// Notify the cell that we have stepped in to it, and whether we are infected
		MPI_Isend(&infected, 1, MPI_INT, cell_proc, SQUIRREL_STEP, comw, &step_send);
		while (!stepped) {
			if (shouldWorkerStop()) break;
			MPI_Test(&step_send, &stepped, MPI_STATUS_IGNORE);
		}
		if (!stepped) break; // We broke due to shouldWorkerStop() so we should stop
		else stepped = 0;

		// Receive the corresponding population and infection levels
		MPI_Recv(&avg_pop, 1, MPI_FLOAT, cell_proc, AVG_POP, comw, &pop_recv);
		MPI_Recv(&avg_inf, 1, MPI_FLOAT, cell_proc, AVG_INF, comw, &inf_recv);

		if (infected) {
			steps_since_inf++;
			if (steps_since_inf > 50) alive = willDie(&state);
		}
		else {
			infected = willCatchDisease(avg_inf, &state);
		}

		if (willGiveBirth(avg_pop, &state)) { // TODO: Check if previous child has received the buffered position first
			new_squirrel = startWorkerProcess();
			x_buf = x; y_buf = y;
			MPI_Isend(&x_buf, 1, MPI_FLOAT, new_squirrel, GET_POSITION, comw, &pos_send);
			MPI_Isend(&y_buf, 1, MPI_FLOAT, new_squirrel, GET_POSITION, comw, &pos_send);
			MPI_Isend(&cells, num_env_cells, MPI_INT, new_squirrel, GET_CELLS, comw, &cell_send);
			if (DEBUG) {
				char* debug_message;
				sprintf(debug_message, "Squirrel gave birth to squirrel on %d", x);
				debug_msg(debug_message);
			}
		}

		if (shouldWorkerStop()) break; // If the simulation has been ended, this worker should stop
	}
}

static void environmentCode(int cell) {
	int current_month = 1, squirrels_this = 0, inf_this = 0, stepped = 0, incomming_inf;
	int squirrels_last1 = 0, squirrels_last2 = 0, inf_last = 0;
	int pop_flux, inf_lev;
	double start = MPI_Wtime();
	
	while (current_month <= max_months) {
		MPI_Request squirrel_step;
		MPI_Status squirrel_step_status;

		// Wait for a squirrel to step on us
		MPI_Irecv(&incomming_inf, 1, MPI_INT, MPI_ANY_SOURCE, SQUIRREL_STEP, comw, &squirrel_step);
		while (!stepped) {
			if (shouldWorkerStop()) break;
			MPI_Test(&squirrel_step, &stepped, &squirrel_step_status);
		}
		if (!stepped) break; // We broke due to shouldWorkerStop() so we should stop
		else stepped = 0;

		squirrels_this++;
		if (incomming_inf) inf_this++;

		pop_flux = squirrels_this + squirrels_last1 + squirrels_last2;
		inf_lev = inf_last + inf_this;

		MPI_Ssend(&pop_flux, 1, MPI_INT, squirrel_step_status.MPI_SOURCE, AVG_POP, comw); // TODO: Make this one, asynchronous message
		MPI_Ssend(&inf_lev, 1, MPI_INT, squirrel_step_status.MPI_SOURCE, AVG_INF, comw);

		// Do some test to see if the month should change
		if (MPI_Wtime() - start > current_month * month_time) {
			printf("Environment Cell %d finished month %d", cell, current_month);
			printf("Pop Influx: %d\tInf Level: %d", pop_flux, inf_lev);
			squirrels_last2 = squirrels_last1;
			squirrels_last1 = squirrels_this;
			squirrels_this = 0;
			inf_last = inf_this;
			inf_this = 0;
			current_month++;
		}

		if (shouldWorkerStop()) break; // If the simulation has been ended, this worker should stop
	}
}
	
