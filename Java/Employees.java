import java.util.*;

public class Employees {
    public static void main(String[] args) {
        new Employees();
    }

    public Employees() {
        Employee[] empArray = {
                new Employee("Jakob", 4000, 1996, 10, 6),
                new Employee("Samuel", 1000, 2000, 31, 10),
                new Employee("Elsa", 2000, 1899, 22, 2),
                new Employee("Elsa", 3000, 1900, 22, 2)
        };
        List<Employee> empList = Arrays.asList(empArray);

        System.out.println("Original List:");
        System.out.println(empList);

        // Sort by Salary
        Collections.sort(empList, new SalaryComparator());
        System.out.println("Sorted by Salary:");
        System.out.println(empList);

        // Sort by Hired Date
        Collections.sort(empList, new HiredDateComparator());
        System.out.println("Sorted by Hired Date:");
        System.out.println(empList);

        // Sort by Name and then Salary
        Collections.sort(empList, new NameComparator());
        System.out.println("Sorted by Name and then Salary:");
        System.out.println(empList);
    }

    public static class Employee {
        private String name;
        private Date hireDate;
        private int salary;

        public Employee(String name, int salary, int year, int month, int day) {
            this.name = name;
            this.salary = salary;
            GregorianCalendar cal = new GregorianCalendar(year, month, day);
            hireDate = cal.getTime();
        }

        public String getName() {
            return this.name;
        }

        public Date getHireDate() {
            return this.hireDate;
        }

        public int getSalary() {
            return this.salary;
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) return false;
            if (o.getClass() != this.getClass()) {
                return false;
            }
            Employee e = (Employee) o;
            return (e.getName().equals(name) && e.getHireDate().equals(hireDate) && e.getSalary() == salary);
        }

        @Override
        public String toString() {
            String newline = "\n";
            return newline + "Employee" + name + newline + "Date of Hire:" + hireDate + newline + "Salary: " + salary + newline;
        }
    }

    public static class SalaryComparator implements Comparator<Employee> {
        @Override
        public int compare(Employee e1, Employee e2) {
            return Integer.compare(e1.getSalary(), e2.getSalary());
        }
    }

    public static class HiredDateComparator implements Comparator<Employee> {
        @Override
        public int compare(Employee e1, Employee e2) {
            return e1.getHireDate().compareTo(e2.getHireDate());
        }
    }

    public static class NameComparator implements Comparator<Employee> {
        @Override
        public int compare(Employee e1, Employee e2) {
            int nameComparison = e1.getName().compareTo(e2.getName());
            if (nameComparison == 0) {
                return Integer.compare(e1.getSalary(), e2.getSalary());
            } else {
                return nameComparison;
            }
        }
    }
}