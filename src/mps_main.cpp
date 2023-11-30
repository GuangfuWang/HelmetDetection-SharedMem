
#ifdef __linux__
#include <sys/sem.h>
#include <unistd.h>
#include <string>
#include "util.h"
#include <sys/shm.h>
#include "model.h"
#include "macro.h"
#else
#pragma message("Only linux platform is supported now.")
#endif


using namespace helmet;

union semun
{
	int val;    /* Value for SETVAL */
	struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
	unsigned short *array;  /* Array for GETALL, SETALL */
	struct seminfo *__buf;  /* Buffer for IPC_INFO
                                           (Linux-specific) */
};

void get_key(int sem_id)// actually the P op.
{
	struct sembuf sop;
	sop.sem_num = 0;
	sop.sem_op = -1;
	sop.sem_flg = SEM_UNDO;
	semop(sem_id, &sop, 1);
}

void back_key(int sem_id)// actually the V op.
{
	struct sembuf sop;
	sop.sem_num = 0;
	sop.sem_op = 1;
	sop.sem_flg = SEM_UNDO;
	semop(sem_id, &sop, 1);
}

int IMG_BYTES = atoi(IMG_BYTE_SIZE);



int main(int argc, char **argv)
{
//	auto pid = static_cast<int>(getpid());
	printf(ANSI_COLOR_GREEN"Num of args: %d\n",argc);
	int pid = 0;
	if (argc < 2) {
		printf(ANSI_COLOR_RED"Wrong input params.\n");
	}else{
		pid = atoi(argv[1]);
	}
	printf(ANSI_COLOR_GREEN"Helmet Worker Process Running...\n");

	std::string stop_shm_addr = "helmet_stop_" + std::to_string(pid);
	std::string rdata_shm_addr = "helmet_pid_" + std::to_string(pid);
	std::string wdata_shm_addr = "helmet_processed_pid_" + std::to_string(pid);

	auto stop_shm_key = ftok("/home", pid);
	auto raw_shm_key = ftok("/home", pid + 100);
	auto p_data_shm_key = ftok("/home", pid + 300);

	//create or get these sems
	// raw_data_key is used for
	int raw_data_key = ftok("/home", pid + 600);
	int raw_data_semid = semget(raw_data_key, 1, IPC_CREAT | 0666);
	int processed_data_key = ftok("/home", pid + 800);
	int processed_data_semid = semget(processed_data_key, 1, IPC_CREAT | 0666);

	bool init = false;
	//this flag will be set by parent process with a semaphore.
	int terminate = 0;
	cvModel *ptr = nullptr;

	int shm_id = shmget(raw_shm_key, IMG_BYTES, IPC_CREAT | 0666);
	int p_shm_id = shmget(p_data_shm_key, IMG_BYTES, IPC_CREAT | 0666);
	auto *raw_data_shm_ptr = (uchar *)shmat(shm_id, 0, SHM_RDONLY);
	auto *p_data_shm_ptr = (uchar *)shmat(p_shm_id, 0, 0);
	int stop_id = shmget(stop_shm_key, sizeof(int), IPC_CREAT | 0666);
	auto *stop_shm_ptr = (int *)shmat(stop_id, 0, SHM_RDONLY);

	printf(ANSI_COLOR_GREEN"IPC Shared Memory Allocation Done...\n");
	int frame_cnt = 0;
	while (!terminate) {
		// get rdata_addr.
		get_key(raw_data_semid);
		cv::Mat img(cv::Size(1920, 1080), CV_8UC3);
//		img.data = raw_data_shm_ptr;
		memcpy(img.data,raw_data_shm_ptr,IMG_BYTES);
		cv::Mat curr = img.clone();
		back_key(raw_data_semid);
		if (!init) {
			ptr = Allocate_Algorithm(curr, IA_TYPE_PEOPLEHELME_DETECTION, 0);
			SetPara_Algorithm(ptr, IA_TYPE_PEOPLEHELME_DETECTION);
			UpdateParams_Algorithm(ptr);
		}
		Process_Algorithm(ptr, curr);

		if(init)get_key(processed_data_semid);
		p_data_shm_ptr = (uchar *)shmat(p_shm_id, NULL, 0);
		memcpy(p_data_shm_ptr, curr.data, IMG_BYTES);
		back_key(processed_data_semid);
		//get termination info.
		//read data.
		if (init) {
			terminate = *stop_shm_ptr;
		}
		init = true;
		frame_cnt++;
//		printf(ANSI_COLOR_GREEN"Current Frame: %d\n",frame_cnt);
	}
	shmdt(raw_data_shm_ptr);
	shmdt(p_data_shm_ptr);
	shmdt(stop_shm_ptr);

	printf(ANSI_COLOR_GREEN"Child process: %d service for helmet detection closed.\n",pid);

}
