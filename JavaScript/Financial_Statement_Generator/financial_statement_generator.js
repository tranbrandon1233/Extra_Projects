const fs = require('fs');

// Transaction validation
function validateTransaction(transaction) {
  const errors = [];
  
  // Check if required details are present
  if (!transaction.amount || !transaction.type || !transaction.date || !transaction.category) {
    errors.push('Missing required transaction details');
    return { valid: false, errors };
  }
  
  // Check if "amount" is a valid number and positive
  const amount = parseFloat(transaction.amount);
  if (isNaN(amount) || amount <= 0) {
    errors.push('Invalid or negative amount');
  }
  
  // Check if the "type" is valid ('credit' or 'debit')
  if (!['credit', 'debit'].includes(transaction.type.toLowerCase())) {
    errors.push('Invalid transaction type. Must be "credit" or "debit"');
  }
  
  // Check if the date is valid
  const parsedDate = new Date(transaction.date);
  if (isNaN(parsedDate.getTime())) {
    errors.push('Invalid date format');
  } else {
    // Only check for future date if the date is valid
    if (parsedDate > new Date()) {
      errors.push('Future-dated transaction detected');
    }
  }
  
  // Check if "category" is a non-empty string
  if (typeof transaction.category !== 'string' || transaction.category.trim() === '') {
    errors.push('Invalid or missing category');
  }
  
  return errors.length > 0 ? { valid: false, errors } : { valid: true };
}

// Transaction processing: calculate summary
function calculateSummary(transactions) {
  return transactions.reduce((summary, transaction) => {
    const amount = parseFloat(transaction.amount);
    if (transaction.type.toLowerCase() === 'credit') {
      summary.income += amount;
    } else if (transaction.type.toLowerCase() === 'debit') {
      summary.expenses += amount;
    }
    summary.balance = Number((summary.income - summary.expenses).toFixed(2));
    summary.income = Number(summary.income.toFixed(2));
    summary.expenses = Number(summary.expenses.toFixed(2));
    return summary;
  }, { income: 0, expenses: 0, balance: 0 });
}

// Filter transactions by month and category
function filterTransactions(transactions, month, category) {
  if (!month && !category) return transactions;

  return transactions.filter(transaction => {
    const transactionDate = new Date(transaction.date);
    let matchesMonth = true;
    let matchesCategory = true;

    if (month) {
      matchesMonth = transactionDate.getMonth() + 1 === parseInt(month);
    }

    if (category) {
      matchesCategory = transaction.category.toLowerCase() === category.toLowerCase();
    }

    return matchesMonth && matchesCategory;
  });
}

// Export filtered transactions to CSV
function exportToCsv(transactions, filePath) {
  if (!transactions || transactions.length === 0) {
    const header = 'Date,Amount,Type,Category\n';
    fs.writeFileSync(filePath, header, 'utf8');
    return true;
  }

  const header = 'Date,Amount,Type,Category\n';
  const csvContent = transactions.map(transaction => {
    const amount = parseFloat(transaction.amount).toFixed(2);
    return `${transaction.date},${amount},${transaction.type},${transaction.category}`;
  }).join('\n');
  
  try {
    fs.writeFileSync(filePath, header + csvContent, 'utf8');
    return true;
  } catch (error) {
    console.error('Error writing CSV file:', error);
    return false;
  }
}

// Format error object for better display
function formatErrorObject(errorObj) {
  return {
    transaction: { ...errorObj.transaction },
    errors: [...errorObj.errors]
  };
}

// Main function to generate report
function generateMonthlyReport(transactions, month, category, filePath) {
  const errors = [];
  const validTransactions = [];
  
  // Validate each transaction
  transactions.forEach(transaction => {
    const validationResult = validateTransaction(transaction);
    if (!validationResult.valid) {
      errors.push(formatErrorObject({
        transaction,
        errors: validationResult.errors
      }));
    } else {
      validTransactions.push({
        ...transaction,
        amount: parseFloat(transaction.amount).toFixed(2),
        type: transaction.type.toLowerCase()
      });
    }
  });
  
  // Filter and process valid transactions
  const filteredTransactions = filterTransactions(validTransactions, month, category);
  const summary = calculateSummary(filteredTransactions);
  
  // Export filtered transactions to CSV
  const exportSuccess = exportToCsv(filteredTransactions, filePath);
  
  // Return comprehensive report with properly formatted errors
  return {
    summary,
    errors: errors.map(error => ({
      transaction: {
        amount: error.transaction.amount,
        type: error.transaction.type,
        date: error.transaction.date,
        category: error.transaction.category
      },
      errors: error.errors
    })),
    transactionCount: {
      total: transactions.length,
      valid: validTransactions.length,
      filtered: filteredTransactions.length
    },
    exportSuccess,
    filterCriteria: {
      month,
      category
    }
  };
}

// Example usage with test transactions
const transactions = [
  { amount: '100.50', type: 'credit', date: '2024-10-11', category: 'Paycheck' },
  { amount: '50.25', type: 'debit', date: '2024-10-15', category: 'Utilities' },
  { amount: '200.75', type: 'credit', date: '2024-10-25', category: 'Paycheck' },
  { amount: '150.00', type: 'debit', date: '2024-10-27', category: 'Rent' },
  { amount: '150.00', type: 'debit', date: '2024-10-029', category: 'Groceries' } 
];

const report = generateMonthlyReport(transactions, '10', 'Paycheck', 'report.csv');
console.log(JSON.stringify(report, null, 2)); // Pretty print the report