using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows.Forms;

public partial class MainForm : Form
{
    private BindingList<PersonData> dataList;
    private List<int> hiddenRows;
    private int lastUsedId = 0;

    public MainForm()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        dataList = new BindingList<PersonData>();
        hiddenRows = new List<int>();

        // Initialize DataGridView
        var dataGridView1 = new DataGridView
        {
            Dock = DockStyle.Top,
            Height = 400,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false
        };

        // Initialize Add Button
        var btnAddNew = new Button
        {
            Text = "Add New Line",
            Dock = DockStyle.Bottom,
            Height = 40
        };
        btnAddNew.Click += BtnAddNew_Click;

        // Setup DataGridView columns
        dataGridView1.AutoGenerateColumns = false;

        dataGridView1.Columns.Add(new DataGridViewTextBoxColumn
        {
            DataPropertyName = "Id",
            HeaderText = "ID",
            Name = "Id",
            ReadOnly = true,
            Width = 50
        });

        dataGridView1.Columns.Add(new DataGridViewTextBoxColumn
        {
            DataPropertyName = "Name",
            HeaderText = "Name",
            Name = "Name",
            Width = 200
        });

        dataGridView1.Columns.Add(new DataGridViewTextBoxColumn
        {
            DataPropertyName = "Age",
            HeaderText = "Age",
            Name = "Age",
            Width = 100
        });

        var deleteButton = new DataGridViewButtonColumn
        {
            HeaderText = "Delete",
            Text = "Delete",
            UseColumnTextForButtonValue = true,
            Width = 80
        };
        dataGridView1.Columns.Add(deleteButton);

        // Setup events
        dataGridView1.CellClick += DataGridView1_CellClick;
        dataGridView1.DataBindingComplete += DataGridView1_DataBindingComplete;

        // Bind data source
        dataGridView1.DataSource = dataList;

        // Add initial data
        AddPerson("John Doe", 30, true);
        AddPerson("Jane Smith", 25, true);
        AddPerson("Bob Johnson", 45, true);
        AddPerson("Alice Brown", 35, true);

        // Form setup
        this.Size = new System.Drawing.Size(800, 500);
        this.Controls.AddRange(new Control[] { dataGridView1, btnAddNew });
    }

    private void AddPerson(string name, int age, bool isInitialData)
    {
        lastUsedId++;
        dataList.Add(new PersonData
        {
            Id = lastUsedId,
            Name = name,
            Age = age,
            IsInitialData = isInitialData
        });
    }

    private void BtnAddNew_Click(object sender, EventArgs e)
    {
        AddPerson($"New Person {lastUsedId + 1}", 0, false);
    }

    private void DataGridView1_CellClick(object sender, DataGridViewCellEventArgs e)
    {
        var dataGridView = (DataGridView)sender;
        if (e.ColumnIndex == dataGridView.Columns.Count - 1 && e.RowIndex >= 0)
        {
            var person = dataList[e.RowIndex];

            if (person.IsInitialData)
            {
                // For initial data, just hide the row
                hiddenRows.Add(e.RowIndex);
                dataGridView.CurrentCell = null;
                dataGridView.Rows[e.RowIndex].Visible = false;
            }
            else
            {
                // For new data, remove from the list
                dataList.RemoveAt(e.RowIndex);
            }
        }
    }

    private void DataGridView1_DataBindingComplete(object sender, DataGridViewBindingCompleteEventArgs e)
    {
        var dataGridView = (DataGridView)sender;
        foreach (int rowIndex in hiddenRows)
        {
            if (rowIndex < dataGridView.Rows.Count)
            {
                dataGridView.CurrentCell = null;
                dataGridView.Rows[rowIndex].Visible = false;
            }
        }
    }
}

public class PersonData
{
    public int Id { get; set; }
    public string Name { get; set; }
    public int Age { get; set; }
    public bool IsInitialData { get; set; }
}