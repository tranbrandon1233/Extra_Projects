import tqdm
import signal

from multiprocessing import Pool

def convert_quality_pool_handler(file_list, convert_quality=1080, convert_quality_process_number=10):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool_convert_quality = Pool(convert_quality_process_number)
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:  
        for _ in tqdm.tqdm(pool_convert_quality.imap_unordered(convert_quality,
            file_list), total=len(file_list)):
            pass
    except KeyboardInterrupt:
        pool_convert_quality.terminate()
    else:
        pool_convert_quality.close()
    finally:
        pool_convert_quality.join()