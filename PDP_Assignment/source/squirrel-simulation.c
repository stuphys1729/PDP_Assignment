/*
*	Author: B059476
*	This program models red squirrels
*/

#include <stdio.h>
#include "mpi.h"
#include "pool.h"

#include "squirrel-functions.h"

#define init_squirrels 1
#define init_infected 0
#define num_env_cells 16
#define equil_steps 20
#define max_months 4
#define month_time 0.1 // How much real time to use as a simulated month (in seconds)
#define squirrel_buffer 50
#define squirrel_birth 0 // Can define whether squirrels can give birth (good for debugging)
#define f_squirrel -1 // Defines a squirrel class, 0 or a positive integer indicates a certain land cell

// MPI Tags
#define FUNCTION_CALL 112
#define GET_POSITION 113
#define GET_CELLS 114
#define MONTH_END 115
#define SIM_START 116
#define SQUIRREL_STEP 123
#define AVG_POP 124
#define AVG_INF 125

#define MASTER 0
#define COORDINATOR 1

#define DEBUG 1
#define VERB_DEBUG 1

static double start_time;
static int rank;

static void workerCode();
static void coordinatorCode();
static void squirrelCode(int, int, int*);
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
		int workerPid = startWorkerProcess();
		if (workerPid != COORDINATOR) {
			char bad_setup[50];
			sprintf(bad_setup, "Corrdinator was started on process %d", rank);
			error_msg(bad_setup);
		}

		int masterStatus = masterPoll();
		while (masterStatus) {
			masterStatus = masterPoll(); // Pass something to tell if a process was killed or started?

		}
		printf("Master is finishing...\n");
	}
	// Finalizes the process pool, call this before closing down MPI
	processPoolFinalise();
	printf("PROCESS %03d CALLED POOLFINALISE\n");
	// Finalize MPI, ensure you have closed the process pool first
	MPI_Finalize();
	printf("PROCESS %03d MADE IT TO THE END\n", rank);
	return 0;
}

static void debug_msg(char* msg) { // TODO: make these functions take arguments like printf
	double time = MPI_Wtime() - start_time;
	printf("[%2.4f] | [Process %03d] | %s\n", time, rank, msg);
}

static void error_msg(char * msg) {
	double time = MPI_Wtime() - start_time;
	fprintf(stderr, "[%2.4f] | [Process %03d] | %s\n", time, rank, msg);
	MPI_Abort(comw, 1);
}

static void workerCode() {
	int workerStatus = 1, function[num_env_cells+2], env_cells[num_env_cells];
	MPI_Status function_stat;

	while (workerStatus) {
		int parent = getCommandData(); // The wake-up data tells us who started us
		
		if (parent == MASTER) { // Master started us, so we are the coordinator
			if (DEBUG) printf("Starting Coordinator\n");
			coordinatorCode();
		}
		else if (parent == COORDINATOR) {// Coordinator started us, so we could be a land cell
			MPI_Recv(&function, num_env_cells+2, MPI_INT, COORDINATOR, FUNCTION_CALL, comw, &function_stat);
			if (function[0] > -1) environmentCode(function[0]); // This is a land cell
			else {
				for (int i = 0; i < num_env_cells; i++) {
					env_cells[i] = function[i + 2];
				}
				squirrelCode(COORDINATOR, function[1], env_cells); // This is one of the initial squirrels
			}
		}
		else {
			// We were started by another squirrel giving birth to us
			squirrelCode(parent, 0, env_cells); // env_cells will be empty, but not used
		}
		workerStatus = workerSleep();	// This MPI process will sleep, further workers may be run on this process now
	}

}

static void coordinatorCode() {
	/*
	* This is going to be the coordinator, keeping track of the month changing and squirrel life-cycles
	*/
	int i, infected, active_squirrels = 0, infected_squirrels = 0, env_cell_ids[num_env_cells + 2];

	// Set some requests for the start of the simulation sync-up
	MPI_Request sim_start[num_env_cells + init_squirrels];

	// Initialise the environment cells
	MPI_Request environment_requests[max_months + 1][num_env_cells];
	for (i = 0; i < num_env_cells; i++) {
		int workerPid = startWorkerProcess();
		env_cell_ids[i+2] = workerPid; // Leaving space for other ints in the message
		env_cell_ids[0] = i; // Just so there is the same size message sent to new actors
		// Tell the processor it's an environmnt cell
		MPI_Isend(&env_cell_ids, num_env_cells+2, MPI_INT, workerPid, FUNCTION_CALL, comw, &environment_requests[0][i]);
		// Set a recv for the simulation start
		MPI_Irecv(NULL, 0, MPI_INT, workerPid, SIM_START, comw, &sim_start[i]);
		// Check when environment cells complete a month
		MPI_Irecv(NULL, 0, MPI_INT, workerPid, MONTH_END, comw, &environment_requests[1][i]); // First index is now month number
		printf("Coordinator started environment cell %d on MPI process %d\n", i, workerPid);
	}

	env_cell_ids[0] = f_squirrel; // Telling processor they are a squirrel
	// Initialise the squirrels
	MPI_Request initial_squirrel_requests[init_squirrels];
	for (i = 0; i<init_squirrels; i++) {
		int workerPid = startWorkerProcess();
		infected = (i < init_infected);
		// Tell the processor it's a squirrel, and if it is infected
		env_cell_ids[1] = infected;
		// And where the environment cells are
		MPI_Isend(&env_cell_ids, num_env_cells+2, MPI_INT, workerPid, FUNCTION_CALL, comw, &initial_squirrel_requests[i]);
		// Then a recv for the simulation start
		MPI_Irecv(NULL, 0, MPI_INT, workerPid, SIM_START, comw, &sim_start[i + num_env_cells]);
		active_squirrels++;
		printf("Coordinator started squirrel %d on MPI process %d\n", i, workerPid);
	}

	// Wait for all the initial processes to be ready
	MPI_Waitall(num_env_cells + init_squirrels, sim_start, MPI_STATUSES_IGNORE);
	start_time = MPI_Wtime(); // Reset the start time so we are mostly in sync

	// Run the simulation
	int current_month = 1, month_end = 0;
	MPI_Status environment_statuses[num_env_cells];

	while (1) {
		// Check if all cells have finished the current month
		MPI_Testall(num_env_cells, environment_requests[current_month], &month_end, environment_statuses);
		if (month_end) {
			printf("[%2.4f] | All cells have completed month %d | ", MPI_Wtime() - start_time, current_month);
			printf("Living Squirrels: %03d\tInfected Squirrels: %03d\n", active_squirrels, infected_squirrels);
			current_month++;
			if (current_month > max_months) {
				shutdownPool();
				break; // Simulation has ended
			}
			// Preapre messages for next month
			for (i = 0; i < num_env_cells; i++) {
				MPI_Irecv(NULL, 0, MPI_INT, env_cell_ids[i+2], MONTH_END, comw, &environment_requests[current_month][i]);
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
		//if (shouldWorkerStop()) break;
	}
	printf("Coordinator is finishing...\n");
}

static void squirrelCode(int parent, int inc_infected, int* inc_cells)
{
	int my_rank, i;
	int* cells = inc_cells;
	float x=0, y=0, x_new, y_new, inc_pos[2];
	MPI_Status cell_recv, pos_recv;

	MPI_Comm_rank(comw, &my_rank);
	long state = -1 - my_rank;

	initialiseRNG(&state); // Initialise random number generation

	int inf_step, infected = inc_infected;
	if (parent == COORDINATOR) {// We are an initial squirrel
		// Could be infected
		if (infected) inf_step = 0;
		// Let the squirrel get to a position independent of the start
		for (i = 0; i < equil_steps; i++) {
			squirrelStep(x, y, &x_new, &y_new, &state);
			x = x_new; y = y_new;
		}
	}
	else { // We have been born by another squirrel
		// Get the ranks of the environment cells
		MPI_Recv(inc_pos, 2, MPI_FLOAT, parent, GET_POSITION, comw, &pos_recv); // TODO: Make this one message for optimisation
		MPI_Recv(&cells, num_env_cells, MPI_INT, parent, GET_CELLS, comw, &cell_recv);
		x = inc_pos[0]; y = inc_pos[1];
	}
	if (DEBUG) {
		char debug_message[50];
		sprintf(debug_message, "Squirrel started with pos: (%1.3f,%1.3f)", x, y);
		debug_msg(debug_message);
	}
	

	// Wait for the simulation to start
	MPI_Ssend(NULL, 0, MPI_INT, COORDINATOR, SIM_START, comw);
	start_time = MPI_Wtime(); // Reset the start time so we are mostly in sync

	// Simulate the squirrel
	int alive = 1, stepped = 0, cell, cell_proc, new_squirrel;
	int step = -1, multiple;
	float avg_pop, avg_inf, x_buf, y_buf, inf_lev[squirrel_buffer] = { 0 }, pop_inf[squirrel_buffer];

	while (alive) {
		MPI_Request pos_send, cell_send, step_send, step_recv;
		MPI_Status pop_recv, inf_recv;
		float inc_levels[2];
		// Do squirrel stuff
		// Step to new position
		squirrelStep(x, y, &x_new, &y_new, &state);
		x = x_new; y = y_new;

		// Find out the corresponding environment cell
		cell = getCellFromPosition(x, y);
		// And the corresponding governing process
		cell_proc = cells[cell];
		if (VERB_DEBUG) {
			char debug_message[50];
			sprintf(debug_message, "Squirrel stepped in cell %02d on proc %03d", cell, cell_proc);
			debug_msg(debug_message);
		}
		// Notify the cell that we have stepped in to it, and whether we are infected
		MPI_Isend(&infected, 1, MPI_INT, cell_proc, SQUIRREL_STEP, comw, &step_send);
		step++;
		multiple = step % squirrel_buffer;

		// Receive the corresponding population and infection levels
		MPI_Irecv(inc_levels, 1, MPI_FLOAT, cell_proc, AVG_POP, comw, &step_recv);
		while (!stepped) {
			if (shouldWorkerStop()) {
				printf("Squirrel is stopping\n");
				MPI_Cancel(&step_send);
				MPI_Cancel(&step_recv);
				break; // If the simulation has been ended, this worker should stop
			}
			MPI_Test(&step_recv, &stepped, &pop_recv);
		}
		if (!stepped) break; // We broke due to shouldWorkerStop() so we should stop
		else stepped = 0;
		
		pop_inf[multiple] = inc_levels[0];
		inf_lev[multiple] = inc_levels[1];

		if (infected) {
			if (step - inf_step > 50) alive = willDie(&state);
		}
		else {
			avg_inf = 0;
			for (i = 0; i < squirrel_buffer; i++) {
				avg_inf += inf_lev[i];
			}
			avg_inf /= squirrel_buffer;

			infected = willCatchDisease(avg_inf, &state);
			if (infected && DEBUG) {
				char* debug_message = "Squirrel became infected";
				debug_msg(debug_message);
			}
		}
		if (multiple == 0 && step != 0 && squirrel_birth) {
			avg_pop = 0;
			for (i = 0; i < squirrel_buffer; i++) {
				avg_pop += pop_inf[i];
			}
			avg_pop /= squirrel_buffer;

			if (willGiveBirth((float)avg_pop, &state)) { // TODO: Check if previous child has received the buffered position first
				new_squirrel = startWorkerProcess();
				inc_pos[0] = x; inc_pos[1] = y;
				MPI_Isend(inc_pos, 1, MPI_FLOAT, new_squirrel, GET_POSITION, comw, &pos_send);
				MPI_Isend(&cells, num_env_cells, MPI_INT, new_squirrel, GET_CELLS, comw, &cell_send);
				if (DEBUG) {
					char debug_message[50];
					sprintf(debug_message, "Squirrel gave birth to squirrel on %03d", new_squirrel);
					debug_msg(debug_message);
				}
			}
		}

		if (shouldWorkerStop()) {
			printf("Squirrel is stopping\n");
			break; // If the simulation has been ended, this worker should stop
		}
	}
	if (!alive && DEBUG) { // Might not have stopped due to death
		char* debug_message = "Squirrel died :( ";
		debug_msg(debug_message);
	}
}

static void environmentCode(int cell) {
	int current_month = 1, incomming_inf, stepped = 0, month_end = 0;
	float squirrels_this = 0.0f, inf_this = 0.0f;
	float squirrels_last1 = 0.0f, squirrels_last2 = 0.0f, inf_last = 0.0f;
	float pop_flux = 0.0f, inf_lev = 0.0f, send_levs[2];
	MPI_Request month_send;

	// Wait for the simulation to start
	MPI_Ssend(NULL, 0, MPI_INT, COORDINATOR, SIM_START, comw);
	double start_time = MPI_Wtime(); // Reset the start time so we are mostly in sync
	
	while (current_month <= max_months) {
		MPI_Request squirrel_step, squirrel_send;
		MPI_Status squirrel_step_status; 

		// Wait for a squirrel to step on us
		MPI_Irecv(&incomming_inf, 1, MPI_INT, MPI_ANY_SOURCE, SQUIRREL_STEP, comw, &squirrel_step);
		if (VERB_DEBUG) {
			char* debug_message = "Waiting for squirrel to step on me";
			debug_msg(debug_message);
		}
		while (!stepped) {
			if (shouldWorkerStop()) break;
			if (MPI_Wtime() - start_time > current_month * month_time) {
				month_end = 1;
				MPI_Cancel(&squirrel_step);
				break;
			}
			MPI_Test(&squirrel_step, &stepped, &squirrel_step_status);
		}
		if (!stepped) { // We broke without being stepped on
			if (!month_end) break; // The simulation has stopped prematurely
		}
		else {
			stepped = 0;

			if (VERB_DEBUG) {
				char debug_message[50];
				sprintf(debug_message, "Squirrel on process %03d stepped on me", squirrel_step_status.MPI_SOURCE);
				debug_msg(debug_message);
			}

			squirrels_this++;
			if (incomming_inf) inf_this++;

			pop_flux = squirrels_this + squirrels_last1 + squirrels_last2;
			inf_lev = inf_last + inf_this;
			send_levs[0] = pop_flux; send_levs[1] = inf_lev;
			MPI_Isend(send_levs, 1, MPI_FLOAT, squirrel_step_status.MPI_SOURCE, AVG_POP, comw, squirrel_send);
		}

		// Do a test to see if the month should change
		if (month_end || MPI_Wtime() - start_time > current_month * month_time) {
			double time = MPI_Wtime() - start_time;
			printf("[%3.4f] | Environment Cell %02d finished month %02d | ", time, cell, current_month);
			printf("Pop Influx: %3.f\tInf Level: %3.f\n", pop_flux, inf_lev);
			if (current_month == max_months) {
				MPI_Ssend(NULL, 0, MPI_INT, COORDINATOR, MONTH_END, comw);
				break;
			} // Last message should be blocking
			squirrels_last2 = squirrels_last1;
			squirrels_last1 = squirrels_this;
			squirrels_this = 0;
			inf_last = inf_this;
			inf_this = 0;
			current_month++;
			month_end = 0;
			MPI_Isend(NULL, 0, MPI_INT, COORDINATOR, MONTH_END, comw, &month_send);
			if (VERB_DEBUG) {
				char msg[50];
				sprintf(msg, "Environment cell %02d sent message for month %02d", cell, current_month-1);
				debug_msg(msg);
			}
		}
	}
	if (DEBUG) {
		char msg[50];
		sprintf(msg, "Environment cell %02d has finished", cell);
		debug_msg(msg);
	}
}
	
