import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.*;
import java.util.HashMap;
import java.util.Map;

public class NoteOrganizerGUI {

    private JFrame frame;
    private JPanel panel;
    private Map<String, String> notes;
    private Map<String, Color> colors;

    public NoteOrganizerGUI() {
        this.notes = new HashMap<>();
        this.colors = new HashMap<>();

        frame = new JFrame("Note Organizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);

        panel = new JPanel();
        panel.setLayout(new GridLayout(0, 3));

        createUI();
        updateUI();

        frame.add(panel);
        frame.setVisible(true);
    }

    private void createUI() {
        JButton addButton = new JButton("Add Note");
        addButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                addNote();
            }
        });

        panel.add(addButton);

        JButton clearButton = new JButton("Clear Notes");
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                clearNotes();
            }
        });

        panel.add(clearButton);

        JButton exitButton = new JButton("Exit");
        exitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.exit(0);
            }
        });

        panel.add(exitButton);
    }

    private void addNote() {
        String category = JOptionPane.showInputDialog("Enter category: ");
        String note = JOptionPane.showInputDialog("Enter note: ");
        Color color = JColorChooser.showDialog(frame, "Select color", Color.WHITE);
        notes.put(category, note);
        colors.put(category, color);
        updateUI();
    }

    private void clearNotes() {
        notes.clear();
        colors.clear();
        updateUI();
    }

    private void updateUI() {
        panel.removeAll();
        createUI();
        for (Map.Entry<String, String> entry : notes.entrySet()) {
            JPanel notePanel = new JPanel();
            notePanel.setBackground(colors.get(entry.getKey()));
            JLabel categoryLabel = new JLabel(entry.getKey());
            JLabel noteLabel = new JLabel(entry.getValue());
            JButton editButton = new JButton("Edit");
            editButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    editNote(entry.getKey());
                }
            });
            notePanel.add(categoryLabel);
            notePanel.add(noteLabel);
            notePanel.add(editButton);
            panel.add(notePanel);
        }
        frame.revalidate();
        frame.repaint();
    }

    private void editNote(String category) {
        String newCategory = JOptionPane.showInputDialog("Enter new category: ");
        String newNote = JOptionPane.showInputDialog("Enter new note: ");
        Color newColor = JColorChooser.showDialog(frame, "Select new color", Color.WHITE);
        notes.put(newCategory, newNote);
        colors.put(newCategory, newColor);
        notes.remove(category);
        colors.remove(category);
        updateUI();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new NoteOrganizerGUI();
            }
        });
    }
}