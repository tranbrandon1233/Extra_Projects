import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class NoteOrganizer {

    private Map<String, String> notes;

    public NoteOrganizer() {
        this.notes = new HashMap<>();
    }

    public void addNote(String category, String note) {
        notes.put(category, note);
        System.out.println("Note added successfully");
    }

    public void viewNotes() {
        for (Map.Entry<String, String> entry : notes.entrySet()) {
            System.out.println("Category: " + entry.getKey() + ", Note: " + entry.getValue());
        }
    }

    public static void main(String[] args) {
        NoteOrganizer organizer = new NoteOrganizer();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("1. Add Note\n2. View Notes\n3. Exit");
            System.out.print("Enter your choice: ");
            int choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    System.out.print("Enter category: ");
                    String category = scanner.next();
                    System.out.print("Enter note: ");
                    String note = scanner.next();
                    organizer.addNote(category, note);
                    break;
                case 2:
                    organizer.viewNotes();
                    break;
                case 3:
                    System.out.println("Exiting...");
                    return;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }
}