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
    
# ... MessageStore class ...

class Client:
    #... init and connect methods ...

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

    # ...close and start_interactive methods...

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

    # ... bind_socket, signal_handler, stop, and start functions ...

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
    # ...other arguments...
    parser.add_argument('-s', '--show', action='store_true', help='Show the most frequently visited webpages for each IP address')
    parser.add_argument('-b', '--ban', help='Ban an IP address')

    return parser.parse_args()

# ...parse_log_file and find_most_frequent_webpages functions...

def find_most_frequent_webpages(webpage_visits):
    """ Find the most frequently visited webpages for each IP address """
    most_frequent_webpages = {}
    for ip_address, webpages in webpage_visits.items():
        counter = Counter(webpages)
        max_count = max(counter.values())
        most_frequent_webpage = min([webpage for webpage, count in counter.items() if count == max_count])
        most_frequent_webpages[ip_address] = (most_frequent_webpage, max_count)
    return most_frequent_webpages

# ...view_messages function...
                
# Main function to parse arguments and run the IP address counter
def main():
    args = parse_args()
    ban_list = BanList()
    
    if args.ban:
        ban_list.ban_ip(args.ban)
        return
    
    # ... other code in main ...
    
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