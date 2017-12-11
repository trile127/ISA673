import socket
import threading
import signal
import sys
import fnmatch
import utils
from time import gmtime, strftime, localtime
import logging
from mla.ml_algorithms_class import MachineLearningAlgorithms
#from mla.ml_algorithms_class_custom import MachineLearningAlgorithms

config =  {
            "HOST_NAME" : "0.0.0.0",
            "BIND_PORT" : 12345,
            "MAX_REQUEST_LEN" : 1024,
            "CONNECTION_TIMEOUT" : 5,
            "BLACKLIST_DOMAINS" : [ "blocked.com" ],
            "HOST_ALLOWED" : [ "*" ],
            "COLORED_LOGGING" : "true"
          }

logging.basicConfig(level=logging.DEBUG,
                    format='[%(CurrentTime)-10s] (%(ThreadName)-10s) %(message)s',
                    )

mla = MachineLearningAlgorithms() # Create ML object

class Server:
    """ The server class """

    def __init__(self, config):
        signal.signal(signal.SIGINT, self.shutdown)     # Shutdown on Ctrl+C
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)             # Create a TCP socket
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # Re-use the socket
        self.serverSocket.bind((config['HOST_NAME'], config['BIND_PORT'])) # bind the socket to a public host, and a port
        self.serverSocket.listen(10)    # become a server socket
        self.__clients = {}


    def listenForClient(self):
        """ Wait for clients to connect """
        while True:
            (clientSocket, client_address) = self.serverSocket.accept()   # Establish the connection
            d = threading.Thread(name=self._getClientName(client_address), target=self.proxy_thread, args=(clientSocket, client_address))
            d.setDaemon(True)
            d.start()
        self.shutdown(0,0)


    def _ishostAllowed(self, host):
        """ Check if host is allowed to access the content """
        for wildcard in config['HOST_ALLOWED']:
            if fnmatch.fnmatch(host, wildcard):
                return True
        return False


    def proxy_thread(self, conn, client_addr):
        """
        *******************************************
        *********** PROXY_THREAD FUNC *************
          A thread to handle request from browser
        *******************************************
        """

        request = conn.recv(config['MAX_REQUEST_LEN'])        # get the request from browser
        first_line = request.split('\n')[0]                   # parse the first line
        url = first_line.split(' ')[1]                        # get url

        # Check if the host:port is blacklisted
        for i in range(0,len(config['BLACKLIST_DOMAINS'])):
            if config['BLACKLIST_DOMAINS'][i] in url:
                self.log("FAIL", client_addr, "BLACKLISTED: " + first_line)
                conn.close()
                # TODO: Create response for 403 Forbidden
                return

        # Check if client is allowed or not
        if not self._ishostAllowed(client_addr[0]):
            # TODO: Create response for 403 Forbidden
            return
	print url
	if (mla.queryurl(url)):
            self.log("FAIL", client_addr, "ML_BLOCKED: " + first_line)
            conn.close()
	    return

        self.log("WARNING", client_addr, "REQUEST: " + first_line)

        # find the webserver and port
        http_pos = url.find("://")          # find pos of ://
        if (http_pos==-1):
            temp = url
        else:
            temp = url[(http_pos+3):]       # get the rest of url

        port_pos = temp.find(":")           # find the port pos (if any)

        # find end of web server
        webserver_pos = temp.find("/")
        if webserver_pos == -1:
            webserver_pos = len(temp)

        webserver = ""
        port = -1
        if (port_pos==-1 or webserver_pos < port_pos):      # default port
            port = 80
            webserver = temp[:webserver_pos]
        else:                                               # specific port
            port = int((temp[(port_pos+1):])[:webserver_pos-port_pos-1])
            webserver = temp[:port_pos]

        try:
            # create a socket to connect to the web server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(config['CONNECTION_TIMEOUT'])
            s.connect((webserver, port))
            s.sendall(request)                           # send request to webserver

            while 1:
                data = s.recv(config['MAX_REQUEST_LEN'])          # receive data from web server
                if (len(data) > 0):
                    conn.send(data)                               # send to browser
                else:
                    break
            s.close()
            conn.close()
        except socket.error as error_msg:
            self.log("ERROR", client_addr, error_msg)
            if s:
                s.close()
            if conn:
                conn.close()
            self.log("WARNING", client_addr, "Peer Reset: " + first_line)


    def _getClientName(self, cli_addr):
        """ Return the clientName.
        """
        return "Client"


    def shutdown(self, signum, frame):
        """ Handle the exiting server. Clean all traces """
        self.log("WARNING", -1, 'Shutting down gracefully...')
        main_thread = threading.currentThread()        # Wait for all clients to exit
        for t in threading.enumerate():
            if t is main_thread:
                continue
            self.log("FAIL", -1, 'joining ' + t.getName())
            t.join()
        self.serverSocket.close()
	mla.removemodels() # Clean models
        sys.exit(0)


    def log(self, log_level, client, msg):
        """ Log the messages to appropriate place """
        LoggerDict = {
            'CurrentTime' : strftime("%a, %d %b %Y %X", localtime()),
            'ThreadName' : threading.currentThread().getName()
        }
        if client == -1:       # Main Thread
            formatedMSG = msg
        else:                  # Child threads or Request Threads
            formatedMSG = '{0}:{1} {2}'.format(client[0], client[1], msg)
        logging.debug('%s', utils.colorizeLog(config['COLORED_LOGGING'], log_level, formatedMSG), extra=LoggerDict)


if __name__ == "__main__":
    server = Server(config)
    server.listenForClient()
