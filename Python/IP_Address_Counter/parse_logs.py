import argparse
from datetime import datetime
import re
from collections import Counter, defaultdict
import socket
import threading
import json
from pathlib import Path
import sys
import signal

MULTICAST_GROUP = '224.0.0.1'
MULTICAST_PORT = 5007

# Class to manage banned IP addresses by loading, adding, and checking banned IPs
class BanList:
    def __init__(self):
        self.store_file = Path('banned_ips.json')  # File to store banned IPs
        self.banned_ips = self._load_bans()
    
    # Load banned IPs from a file if it exists
    def _load_bans(self):
        if self.store_file.exists():
            with open(self.store_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    # Ban an IP by adding it to the list and updating the file
    def ban_ip(self, ip):
        self.banned_ips.add(ip)
        with open(self.store_file, 'w') as f:
            json.dump(list(self.banned_ips), f)
        print(f"Banned IP: {ip}")
    
    # Check if an IP is banned
    def is_banned(self, ip):
        return ip in self.banned_ips
    
class MessageStore:
    def __init__(self):
        self.store_file = Path('message_history.json')
        self.messages = self._load_messages()
        self._lock = threading.Lock()
    
    def _load_messages(self):
        if self.store_file.exists():
            with open(self.store_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_message(self, ip_address, message, timestamp=None):
        with self._lock:
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if ip_address not in self.messages:
                self.messages[ip_address] = []
            
            self.messages[ip_address].append({
                'message': message,
                'timestamp': timestamp,
                'status': 'sent'
            })
            
            with open(self.store_file, 'w') as f:
                json.dump(self.messages, f, indent=2)
    
    def get_messages(self, ip_address=None):
        with self._lock:
            if ip_address:
                return self.messages.get(ip_address, [])
            return self.messages

class Client:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.sock = None
        self.message_store = MessageStore()
        self.running = False
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect(('localhost', 8080))
            self.running = True
            
            self.receive_thread = threading.Thread(target=self.receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False

    def receive_messages(self):
        while self.running and self.sock:
            try:
                data = self.sock.recv(1024).decode()
                if data == "BANNED":
                    print("Sorry, you have been banned")
                    self.running = False
                    break
                if not data:
                    break
                    
                payload = json.loads(data)
                print(f"\nMessage from {payload['sender']}: {payload['message']}")
                print("Enter messages in format '[IP]: [message]' (or 'quit' to exit)", end='', flush=True)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error receiving message: {str(e)}")
                break
        self.close()

    def send_message(self, message):
        if not self.sock or not self.running:
            print("Not connected to server")
            return False
            
        try:
            payload = json.dumps({
                'sender_ip': self.ip_address,
                'message': message
            })
            self.sock.sendall(payload.encode())
            response = self.sock.recv(1024).decode()
            if response == "BANNED":
                print("Sorry, you have been banned")
                self.running = False
                return False
            elif response == "ACK":
                return True
        except Exception as e:
            print(f"Error sending message: {str(e)}")
            self.running = False
            return False
        return False

    def close(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def start_interactive(self):
        if not self.connect():
            return

        print(f"Connected as {self.ip_address}. Enter 'quit' to exit.")
        try:
            while self.running:
                msg = input("Enter message: ")
                if msg.lower() == 'quit':
                    break
                if not self.send_message(msg):
                    break
        finally:
            self.close()

# Server class to manage clients, broadcast messages, and process banning
class Server:
    def __init__(self):
        self.sock = None
        self.message_store = MessageStore()
        self.clients = {}
        self.running = False
        self._lock = threading.Lock()
        self.ban_list = BanList()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.bind_socket()

    def bind_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        PORT = 8080
        while True:
            try:
                self.sock.bind(('localhost', PORT))
                break
            except OSError:
                PORT += 1
        self.sock.listen()
        print(f"Server started on localhost:{PORT}")

    def signal_handler(self, signum, frame):
        print("\nShutting down server...")
        self.stop()
        sys.exit(0)

    def stop(self):
        self.running = False
        # Close all client connections
        with self._lock:
            for conn in self.clients.values():
                try:
                    conn.close()
                except:
                    pass
            self.clients.clear()
        
        # Close server socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass

    def start(self):
        self.running = True
        input_thread = threading.Thread(target=self.handle_user_input)
        input_thread.daemon = True
        input_thread.start()
        
        try:
            while self.running:
                try:
                    self.sock.settimeout(1.0)  # Allow checking self.running periodically
                    conn, addr = self.sock.accept()
                    client_thread = threading.Thread(target=self.handle_client, args=(conn, addr[0]))
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:  # Only print error if not intentionally stopped
                        print(f"Error accepting connection: {str(e)}")
        finally:
            self.stop()

    def handle_user_input(self):
        print("Enter messages in format '[IP]: [message]' (or 'quit' to exit)")
        while self.running:
            try:
                user_input = input()
                if user_input.lower() == 'quit':
                    self.running = False
                    print("Shutting down server...")
                    break
                
                try:
                    ip, message = user_input.split(':', 1)
                    ip = ip.strip()
                    message = message.strip()
                    
                    if self.ban_list.is_banned(ip):
                        print(f"Sorry, {ip} has been banned")
                        continue
                    
                    self.message_store.save_message(ip, message)
                    self.broadcast_message(ip, message)
                except ValueError:
                    print("Invalid format. Use '[IP]: [message]'")
            except EOFError:
                break
            except Exception as e:
                print(f"Error processing input: {str(e)}")

    def broadcast_message(self, sender_ip, message):
        with self._lock:
            dead_clients = []
            for ip, conn in self.clients.items():
                if ip != sender_ip and not self.ban_list.is_banned(ip):
                    try:
                        payload = json.dumps({
                            'sender': sender_ip,
                            'message': message
                        })
                        conn.sendall(payload.encode())
                    except:
                        dead_clients.append(ip)
            
            # Remove dead connections
            for ip in dead_clients:
                del self.clients[ip]

    def handle_client(self, conn, client_ip):
        try:
            if self.ban_list.is_banned(client_ip):
                conn.sendall("BANNED".encode())
                return
            with self._lock:
                self.clients[client_ip] = conn
            
            while self.running:
                try:
                    conn.settimeout(1.0)  # Allow checking self.running periodically
                    data = conn.recv(1024).decode()
                    if not data:
                        break
                        
                    payload = json.loads(data)
                    sender_ip = payload.get('sender_ip', client_ip)
                    message = payload['message']
                    
                    if not self.ban_list.is_banned(sender_ip):
                        print(f"Message from {sender_ip}: {message}")
                        self.message_store.save_message(sender_ip, message)
                        self.broadcast_message(sender_ip, message)
                        conn.sendall("ACK".encode())
                    else:
                        conn.sendall("BANNED".encode())
                        continue
                except Exception as e:
                    if self.running:  # Only print error if not intentionally stopped
                        print(f"Error handling client {client_ip}: {str(e)}")
                    break
        finally:
            with self._lock:
                if client_ip in self.clients:
                    del self.clients[client_ip]
            conn.close()
            
            
# Parse arguments sent from the command line   
def parse_args():
    parser = argparse.ArgumentParser(description='Parse log file and find the most active IP addresses')
    parser.add_argument('n', type=int, nargs='?', default=None, help='Number of most active IP addresses to find')
    parser.add_argument('-d', '--date', help='Specific date in "MM/DD/YYYY" format')
    parser.add_argument('-m', '--month', help='Specific month in "MM" format')
    parser.add_argument('-y', '--year', help='Specific year in "YYYY" format')
    parser.add_argument('-s', '--show', action='store_true', help='Show the most frequently visited webpages for each IP address')
    parser.add_argument('-b', '--ban', help='Ban an IP address')
    parser.add_argument('-f', '--file', default='logs.txt', help='Specify the log file (default: logs.txt)')
    parser.add_argument('--server', action='store_true', help='Run as server')
    parser.add_argument('--view-messages', action='store_true', help='View message history')
    parser.add_argument('--ip', help='View messages for specific IP address')
    return parser.parse_args()

def parse_log_file(log_file, date=None, month=None, year=None):
    """ Parse the log file and extract IP addresses and webpage visits """
    ip_addresses = []
    webpage_visits = defaultdict(list)
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', line)
            if match:
                ip_address = match.group()
                log_date = re.search(r'\[(.*?)\]', line).group(1).split(':')[0]
                log_date = datetime.strptime(log_date, '%d/%b/%Y')
                if (date and log_date.strftime('%m/%d/%Y') == date) or \
                   (month and log_date.strftime('%m') == month) or \
                   (year and log_date.strftime('%Y') == year) or \
                   (not date and not month and not year):
                    ip_addresses.append(ip_address)
                    webpage = re.search(r'"GET (.*?) HTTP', line).group(1)
                    webpage_visits[ip_address].append(webpage)
    return ip_addresses, webpage_visits


def find_most_active_ip_addresses(ip_addresses, n):
    """ Find the most active IP addresses from a list of IP addresses """
    counter = Counter(ip_addresses)
    return counter.most_common(min(n, len(counter)))

def find_most_frequent_webpages(webpage_visits):
    """ Find the most frequently visited webpages for each IP address """
    most_frequent_webpages = {}
    for ip_address, webpages in webpage_visits.items():
        counter = Counter(webpages)
        max_count = max(counter.values())
        most_frequent_webpage = min([webpage for webpage, count in counter.items() if count == max_count])
        most_frequent_webpages[ip_address] = (most_frequent_webpage, max_count)
    return most_frequent_webpages

def view_messages(ip_address=None):
    """ View messages for a specific IP address or all IP addresses """
    message_store = MessageStore()
    messages = message_store.get_messages(ip_address)
    
    if ip_address:
        if not messages:
            print(f"No messages found for {ip_address}")
            return
        print(f"\nMessages for {ip_address}:")
        for msg in messages:
            print(f"[{msg['timestamp']}] {msg['message']}")
    else:
        if not messages:
            print("No messages found")
            return
        print("\nAll messages:")
        for ip, msg_list in messages.items():
            print(f"\n{ip}:")
            for msg in msg_list:
                print(f"[{msg['timestamp']}] {msg['message']}")
                
# Main function to parse arguments and run the IP address counter
def main():
    args = parse_args()
    ban_list = BanList()
    
    if args.ban:
        ban_list.ban_ip(args.ban)
        return
    
    if args.server:
        server = Server()
        server.start()
        return
        
    if args.view_messages:
        view_messages(args.ip)
        return
        
    log_file = args.file
    
    if args.n:
        try:
            ip_addresses, webpage_visits = parse_log_file(log_file, args.date, args.month, args.year)
            most_active_ip_addresses = find_most_active_ip_addresses(ip_addresses, args.n)
                        
            if args.show:
                most_frequent_webpages = find_most_frequent_webpages(webpage_visits)
                for ip_address, count in most_active_ip_addresses:
                    most_frequent_webpage, webpage_count = most_frequent_webpages[ip_address]
                    print(f'{ip_address}: {count}, {most_frequent_webpage}: {webpage_count}')
            else:
                for ip_address, count in most_active_ip_addresses:
                    print(f'{ip_address}: {count}')
        except FileNotFoundError:
            print(f"Error: Log file '{log_file}' not found")
            return
    


if __name__ == '__main__':
    main()