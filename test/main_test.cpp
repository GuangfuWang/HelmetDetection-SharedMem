#include <thread>
#include <future>
#include <sys/sem.h>
#include <sys/shm.h>
#include <unistd.h>
#include "../src/config.h"
#include "../src/trt_deploy.h"
#include "../src/model.h"

using namespace helmet;

int TEST_THREADS = 3;

int print_cnt = 50;

int img_byte_size = atoi(IMG_BYTE_SIZE);

union semun
{
	int val;    /* Value for SETVAL */
	struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
	unsigned short *array;  /* Array for GETALL, SETALL */
	struct seminfo *__buf;  /* Buffer for IPC_INFO
                                           (Linux-specific) */
};

void get_key(int sem_id)
{
	struct sembuf sop;
	sop.sem_num = 0;
	sop.sem_op = -1;
	sop.sem_flg = SEM_UNDO;
	semop(sem_id, &sop, 1);
}

void back_key(int sem_id)
{
	struct sembuf sop;
	sop.sem_num = 0;
	sop.sem_op = 1;
	sop.sem_flg = SEM_UNDO;
	semop(sem_id, &sop, 1);
}

void process_frame(std::mutex &mtx, std::condition_variable &cond,
				   std::queue<cv::Mat> &q, std::queue<cv::Mat> &out,
				   bool &init, volatile bool &running, int stream_num)
{
	std::string stop_shm_addr = "helmet_stop_" + std::to_string(stream_num);
	std::string rdata_shm_addr = "helmet_pid_" + std::to_string(stream_num);
	std::string wdata_shm_addr = "helmet_processed_pid_" + std::to_string(stream_num);

	auto stop_shm_key = ftok("/home", stream_num);
	auto raw_shm_key = ftok("/home", stream_num + 100);
	auto p_data_shm_key = ftok("/home", stream_num + 300);

	//create or get these sems
	// raw_data_key is used for
	int raw_sem_key = ftok("/home", stream_num + 600);
	int raw_data_semid = semget(raw_sem_key, 1, IPC_CREAT | 0666);
	int processed_sem_key = ftok("/home", stream_num + 800);
	int processed_data_semid = semget(processed_sem_key, 1, IPC_CREAT | 0666);

	union semun set1;
	set1.val = 1;
	semctl(raw_data_semid, 0, SETVAL, set1);
	union semun set2;
	set2.val = 0;
	semctl(processed_data_semid, 0, SETVAL, set2);

	auto child = fork();
	if (child < 0) {
		std::cerr << "Thread: " << std::this_thread::get_id() <<
				  " Create the child process failed." << std::endl;
		return;
	}
	else if (child == 0) {
		// this part is inside the child process.
		printf("Child process running...\n");
		std::string exe_file = "./" + std::string(DEPLOY_MPS);
		std::string id = std::to_string(stream_num);
		printf("Worker exe: %s\n", exe_file.c_str());
		execl(exe_file.c_str(), DEPLOY_MPS, id.c_str());
		init = true;
	}
	else {
		int shm_id = shmget(raw_shm_key, img_byte_size, IPC_CREAT | 0666);
		int p_shm_id = shmget(p_data_shm_key, img_byte_size, IPC_CREAT | 0666);
		uchar *raw_data_shm_ptr = (uchar *)shmat(shm_id, 0, 0);;
		uchar *p_data_shm_ptr = nullptr;
		int stop_id = shmget(stop_shm_key, sizeof(int), IPC_CREAT | 0666);
		auto *stop_shm_ptr = (int *)shmat(stop_id, 0, 0);
		memset(stop_shm_ptr, 0, sizeof(int));
		// this part is inside the parent process.
		while (running) {
			std::unique_lock<std::mutex> lk(mtx);
			cond.wait(lk, [&]
			{
				return !q.empty();
			});
			auto img = q.front();
			q.pop();
//			printf("Processing from main_test\n");
			get_key(raw_data_semid);
			memcpy(raw_data_shm_ptr, img.data, img_byte_size);
			back_key(raw_data_semid);
//			printf("Copied the data to shm\n");
			cv::Mat processed = cv::Mat(img.size(), img.type());

			get_key(processed_data_semid);
			p_data_shm_ptr = (uchar *)shmat(p_shm_id, 0, SHM_RDONLY);
			memcpy(processed.data, p_data_shm_ptr, img_byte_size);
//			printf("Get processed data from worker\n");
			back_key(processed_data_semid);

			out.push(processed);
			lk.unlock();
			cond.notify_one();
		}
		//set the termination info.
		memset(stop_shm_ptr, 1, sizeof(int));
		//wait like 10ms for the child process finished.
		usleep(10000);

//		printf("Terminate the frame process thread...\n");
		shmdt(raw_data_shm_ptr);
		shmdt(p_data_shm_ptr);
		shmdt(stop_shm_ptr);
	}

}

void process_video(int thread_id, const std::string &file)
{
	std::mutex m_thread_mtx;
	std::condition_variable m_thread_cond;
	std::queue<cv::Mat> m_data_queue;
	std::queue<cv::Mat> m_out_q;
	bool init = false;
	volatile bool run = true;
	std::thread m_thread = std::thread(process_frame, std::ref(m_thread_mtx),
									   std::ref(m_thread_cond),
									   std::ref(m_data_queue), std::ref(m_out_q),
									   std::ref(init), std::ref(run), thread_id);
	//prepare the input data.
	if (!checkFileExist(file)) {
		std::cerr << "The video file is not exist..." << std::endl;
		return;
	}
	auto in_path = std::filesystem::path(file);
	cv::VideoCapture cap(in_path);
	cv::VideoWriter vw;

	std::filesystem::path
		output_path = in_path.parent_path() / (in_path.stem().string() + std::to_string(thread_id) + ".mp4");
	vw.open(output_path,
			cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
			cap.get(cv::CAP_PROP_FPS),
			cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
					 cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
	int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
	cv::Mat img;
	int f_cnt = 0;
	std::chrono::high_resolution_clock::time_point curr_time;
	while (cap.isOpened()) {
		cap >> img;
		f_cnt++;
		if (f_cnt > total_frames) {
			run = false;
			printf(ANSI_COLOR_BLUE"Stopping the %d-th threads...\n",thread_id);
			break;
		}
		curr_time =
			std::chrono::high_resolution_clock::now();
		if (!img.empty()) {
			std::unique_lock<std::mutex> lock(m_thread_mtx);
			m_data_queue.push(img);
			while (m_out_q.size()) {
				vw.write(m_out_q.front());
				m_out_q.pop();
			}
			lock.unlock();
			m_thread_cond.notify_one();
		}
		auto dur = std::chrono::high_resolution_clock::now() - curr_time;
		curr_time = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		if (f_cnt % print_cnt == 0){
			std::cout << "Thread: " << std::this_thread::get_id() << " Cpu: " <<
					  sched_getcpu() << " taken: " << ms << "ms"
					  << std::endl;
		}
	}
	cap.release();
	vw.release();
	m_thread.join();
}

int main(int argc, char **argv)
{
	bool enable_cpu_affinity = false;
	if (argc > 1) {
		TEST_THREADS = std::atoi(argv[1]);
	}
	if (argc > 2) {
		int temp = std::atoi(argv[2]);
		if (temp)enable_cpu_affinity = true;
	}
	if (argc > 3) {
		enable_trt_log = true;
		enable_config_log = true;
	}
	std::vector<std::thread> threads(TEST_THREADS);
	std::vector<std::string> files(TEST_THREADS);
	std::string base = "/home/wgf/Downloads/datasets/Anquanmao/helmet-live/";
	for (int i = 0; i < TEST_THREADS; ++i) {
//		files[i] = base + std::to_string(i) + ".mp4";
		files[i] = "/home/wgf/Downloads/datasets/Anquanmao/helmet-live/multithread.mp4";
		threads[i] = std::thread(process_video, i, files[i]);
		if (enable_cpu_affinity) {
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(i, &cpuset);
			int rc = pthread_setaffinity_np(threads[i].native_handle(),
											sizeof(cpu_set_t), &cpuset);
			if (rc != 0) {
				std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
			}
		}
	}
	for (int i = 0; i < TEST_THREADS; ++i) {
		threads[i].join();
	}

	return 0;
}