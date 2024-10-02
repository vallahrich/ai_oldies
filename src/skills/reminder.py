import datetime

class ReminderSkill:
    def __init__(self):
        self.reminders = []

    def set_reminder(self, task, time):
        self.reminders.append({"task": task, "time": time})
        return f"Okay, I'll remind you to {task} at {time}."

    def check_reminders(self):
        current_time = datetime.datetime.now().strftime("%H:%M")
        due_reminders = [r for r in self.reminders if r["time"] == current_time]
        return due_reminders
