import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Student {
    String name;

    public Student(String name) {
        if (name == null) {
            throw new IllegalArgumentException("Name cannot be null");
        }
        this.name = name;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Student student = (Student) obj;
        return name.equals(student.name);
    }
}

class StudentContext {

    private final Student student;
    private static final ThreadLocal<StudentContext> threadLocalContext = new ThreadLocal<>();

    private StudentContext(Student student) {
        if (student == null) {
            throw new IllegalArgumentException("Student cannot be null");
        }
        this.student = student;
    }

    public static StudentContext getInstance() {
        StudentContext context = threadLocalContext.get();
        if (context == null) {
            throw new IllegalStateException("Student context not initialized. Please set a student first.");
        }
        return context;
    }

    public static void setStudent(Student student) {
        if (student == null) {
            throw new IllegalArgumentException("Student cannot be null");
        }
        threadLocalContext.set(new StudentContext(student));
    }

    public Student getStudent() {
        return student;
    }

    public static void clear() {
        threadLocalContext.remove();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        StudentContext that = (StudentContext) obj;
        return student.equals(that.student);
    }
}

class StudentContextTest {
    @Test
    void testGetInstance() {
        Student student1 = new Student("Alice");
        StudentContext.setStudent(student1);
        StudentContext context1 = StudentContext.getInstance();
        assertEquals(student1, context1.getStudent());

        Student student2 = new Student("Bob");
        StudentContext.clear(); // Clear the previous context
        assertThrows(IllegalStateException.class, StudentContext::getInstance); // Context should be null now
        StudentContext.setStudent(student2); // Initialize context again
        StudentContext context2 = StudentContext.getInstance();
        assertEquals(student2, context2.getStudent());
    }

    @Test
    void testClear() {
        Student student = new Student("Alice");
        StudentContext.setStudent(student);
        StudentContext context = StudentContext.getInstance();
        assertNotNull(context);
        StudentContext.clear();
        assertThrows(IllegalStateException.class, StudentContext::getInstance);
    }

    @Test
    void testSetStudent_nullStudent() {
        assertThrows(IllegalArgumentException.class, () -> StudentContext.setStudent(null));
    }

    @Test
    void testConstructor_nullStudent() {
        assertThrows(IllegalArgumentException.class, () -> new StudentContext(null));
    }
}