#include <iostream>
#include <string>
#include <vector>

using namespace std;
struct node{
	node(string, int, node*, node*, node*);
	string character;
	int occurence;
	node* left;
	node* right;
	node* next;
};

node::node(string c, int o=1, node* r=0, node* l=0, node* n=0) {
	character = c;
	occurence = o;
	right=r;
	left=l;
	next=n;
}
struct datanode {
	datanode(string,string,datanode*);
	string character;
	string location; 
	datanode* next;
};

datanode::datanode(string c, string l, datanode* n=0) {
	character = c;
	location = l;
	next = n;
}

void insert(node* &nd, string c){ 
	if (nd == 0) { 
		nd=new node(c);
		return;
	}
	else if (c == nd -> character) {
		nd -> occurence += 1;
	}
	else {
		insert(nd -> next, c);
	}
}

bool sort(node* &nd, bool sorted) {
	if (nd -> next == 0) {
		return true;
	}
	if ((nd -> occurence < nd -> next -> occurence) or ((nd -> character > nd -> next -> character) and (nd -> occurence == nd -> next -> occurence))) {
		node* tempnode = nd;
		nd = nd -> next;
		tempnode -> next = nd -> next;
		nd -> next = tempnode;
		sorted = false;
	}

	sorted = sort(nd -> next, sorted);
	return sorted;
}	

bool compress(node* &nd) {
	if (nd -> next == 0) {
		return true;
	}
	else if (nd -> next -> next == 0) { 
		node* treenode = new node(nd->character+nd -> next -> character, nd->occurence + nd -> next -> occurence);
		if (nd -> occurence <= nd -> next -> occurence) { 
			treenode -> right = nd -> next;
			nd -> next = 0;
			treenode -> left = nd;
		}
		
		else {
			treenode -> left = nd -> next;
			nd -> next = 0;
			treenode -> right = nd;		
		}
		
		nd = treenode;
		return false;
	}		
	else {
		compress(nd -> next);
		return false;
	}
}

void clear(node* &root) {
	if (root!=0) {
		clear(root -> left);
		clear(root -> right);
		delete root;
		root=0;
	}
}

void tableinsert(datanode* &nd, string c, string b){ 
	if (nd == 0) { 
		nd=new datanode(c,b);
		return;
	}
	else if ((b.length() < nd -> location.length()) or ((b.length() == nd -> location.length()) and (c < nd -> character))) {
		string tempc = c;
		string tempb = b;
		c = nd -> character;
		b = nd -> location;
		nd -> character = tempc;
		nd -> location = tempb;
		
	}
	tableinsert(nd -> next, c, b);

}
void datatable(node* root, datanode* &dataroot, string binary) {
	if (root -> left != 0) {
		datatable(root -> left, dataroot, binary+"0");
	}
	if (root -> right != 0) {
		datatable(root -> right, dataroot, binary+"1");
	}
	if ((root -> left == 0) and (root -> right == 0)) {
		tableinsert(dataroot, root -> character, binary);
	}
}
void walk(datanode* nd){
	if (nd==0) return;
	cout<< nd -> character <<": " << nd -> location;
	cout << endl;
	walk(nd->next);
}
void popfront(datanode* &nd){
	if (nd !=0) {
		datanode* temp=nd -> next;
		delete nd;
		nd = temp;
	}
}
void lfree(datanode* &nd){
	while (nd!=0) {
		popfront(nd);
	}
}
			
int main() {
	node* root = 0;
	string str = "HEELLO";
	for (unsigned int i=0; i<str.length(); ++i)
	{
		insert(root, string(1, str.at(i)));
	}
	
	bool compressed = false; 
	bool bubblesort;
	while (!compressed) {
		compressed = compress(root);
		bubblesort=false;
		while (bubblesort == false) {
			bubblesort = sort(root,bubblesort);
		}
	}

	datanode* dataroot=0;
	string binary("");
	datatable(root, dataroot, binary);
	walk(dataroot);
	clear(root);
	lfree(dataroot);
	return 0;
}