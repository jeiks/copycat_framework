import logging
import io

class Logger:
    '''
    This class aims to provide a log file for the Copycat process, which
    includes Oracle's training, Copycat's training, etc.
    Example:
        log = Logger()
        log.set_filename('output_file.log')
        log.warning('Be careful')
        log.info('Everyting is ok')
        log.debug('Some bug is here...')
        log.set_filename('output_file2.log')
        log.info('continue logging in other file')
    Example of how to use this class on tqdm:
        from tqdm import tqdm
        import time
        for _ in tqdm(range(100), file=Logger()):
            time.sleep(0.2)
    '''
    _instance = None
    _buf = ''
    filename = None
    _print = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_filename(self, filename='output.log'):
        self.filename = filename
        self.setup()

    def setup(self):
        root = logging.getLogger()
        for h in root.handlers:
            root.removeHandler(h)
            h.close()
        log_format = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y,%Y %H:%M:%S' )
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(filename=self.filename, mode='a')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.INFO)

        root.addHandler(stream_handler)
        root.addHandler(file_handler)
        root.setLevel(logging.INFO)

        logging.info('Date format: D/M/Y')

    def warning(self, msg):
        if self.filename is None: self.set_filename()
        logging.info(f'WARNING: {msg}')
    
    def info(self, msg):
        if self.filename is None: self.set_filename()
        logging.info(msg)
    
    def debug(self, msg):
        if self.filename is None: self.set_filename()
        logging.info(f'DEBUG: {msg}')
    
    def error(self, msg):
        if self.filename is None: self.set_filename()
        logging.info(f'ERROR: {msg}')
    
    def critical(self, msg):
        if self.filename is None: self.set_filename()
        logging.info(f'CRITICAL: {msg}')

    def log(self, msg):
        self.info(msg)
    
    def write(self, buf):
        self._buf = buf

    def flush(self):
        self.info(self._buf.strip('\r\n\t '))