"""Draw the updated use case diagram and the decision-support workflow (matplotlib)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, Circle

OUT = Path(__file__).resolve().parents[1] / "assets"


def actor(ax, x, y, label):
    ax.add_patch(Circle((x, y + 0.55), 0.18, fill=False, lw=1.5))
    ax.plot([x, x], [y + 0.37, y - 0.2], "k", lw=1.5)
    ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], "k", lw=1.5)
    ax.plot([x, x - 0.25], [y - 0.2, y - 0.7], "k", lw=1.5)
    ax.plot([x, x + 0.25], [y - 0.2, y - 0.7], "k", lw=1.5)
    ax.text(x, y - 1.0, label, ha="center", va="top", fontsize=8.5)


def use_case_diagram():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 14); ax.set_ylim(-1.2, 9); ax.axis("off")
    ax.add_patch(Rectangle((3.4, -0.9), 7.2, 9.4, fill=False, lw=1.8))
    ax.text(7, 8.25, "Campus Facility Reservation Monitoring System", ha="center", fontsize=11, weight="bold")
    cases = [
        ("UC-01 Submit Booking Details", 7.3),
        ("UC-02 Assess No-Show Risk (FR-DM-03)", 6.1),
        ("UC-03 Prompt Staff to Confirm (FR-DM-02)", 4.9),
        ("UC-04 Review Risk Queue & Record\nDecision (FR-DM-05)", 3.6),
        ("UC-05 View Prediction Explanation\n& Model Metrics", 2.3),
        ("UC-06 Upload & Validate Dataset (FR-DM-04)", 1.1),
        ("UC-07 Explore Facility Load (FR-DM-01)", -0.1),
    ]
    pos = {}
    for name, y in cases:
        ax.add_patch(Ellipse((7, y), 5.6, 0.95, fill=False, lw=1.4))
        ax.text(7, y, name, ha="center", va="center", fontsize=8.3)
        pos[name[:5]] = y
    actor(ax, 1.4, 6.6, "Reserver\n(Student / Faculty /\nStaff / External)")
    actor(ax, 1.4, 3.0, "Predictive Engine\n(system actor)")
    actor(ax, 12.6, 3.6, "Facility Admin")
    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], "k", lw=1.1)
    line(1.8, 6.9, 4.2, pos["UC-01"])
    line(1.8, 3.4, 4.2, pos["UC-02"])
    line(1.8, 3.2, 4.2, pos["UC-03"])
    for uc in ["UC-04", "UC-05", "UC-06", "UC-07"]:
        line(12.2, 3.9, 9.8, pos[uc])
    def dashed(y1, y2, text, x=7.0):
        ax.annotate("", xy=(x, y2 + 0.48), xytext=(x, y1 - 0.48),
                    arrowprops=dict(arrowstyle="->", ls="--", lw=1.1))
        ax.text(x + 0.1, (y1 + y2) / 2, text, fontsize=7.5, style="italic", va="center")
    dashed(7.3, 6.1, "<<include>>")
    ax.annotate("", xy=(7.0, 6.1 - 0.48), xytext=(7.0, 4.9 + 0.48),
                arrowprops=dict(arrowstyle="->", ls="--", lw=1.1))
    ax.text(7.1, 5.5, "<<extend>> [High-risk Staff]", fontsize=7.5, style="italic", va="center")
    ax.annotate("", xy=(7.0, 2.3 + 0.48), xytext=(7.0, 3.6 - 0.48),
                arrowprops=dict(arrowstyle="->", ls="--", lw=1.1))
    ax.text(7.1, 2.95, "<<include>>", fontsize=7.5, style="italic", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "use_case_diagram.png", dpi=170)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
    def box(x, y, w, h, text, fc="#e8f1fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#33506b", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, fontsize=7.5, ha="center")
    box(0.2, 2.6, 1.9, 1.0, "Reservation\nsubmitted\n(booking-time data)")
    box(2.6, 2.6, 1.9, 1.0, "Validate inputs\n& schema")
    box(5.0, 2.6, 2.1, 1.0, "Saved pipeline\npredicts P(Not Completed)\n(no retraining)")
    box(7.6, 2.6, 1.7, 1.0, "Assign risk tier\nHIGH >= 50%\nMED >= 35%", fc="#fff4d6")
    box(10.2, 4.6, 3.4, 0.9, "LOW: standard workflow,\nno action", fc="#e3f4e1")
    box(10.2, 3.1, 3.4, 0.9, "MEDIUM: automated reminder\n(48 h before slot)", fc="#fff4d6")
    box(10.2, 1.6, 3.4, 0.9, "HIGH: enters admin review queue\n(Staff: confirmation prompt)", fc="#fbe0dc")
    box(10.2, 0.2, 3.4, 0.9, "Facility Admin records FINAL decision\n(logged with model version)", fc="#dfe7f5")
    arrow(2.1, 3.1, 2.6, 3.1); arrow(4.5, 3.1, 5.0, 3.1); arrow(7.1, 3.1, 7.6, 3.1)
    arrow(9.3, 3.3, 10.2, 5.0); arrow(9.3, 3.1, 10.2, 3.55); arrow(9.3, 2.9, 10.2, 2.05)
    arrow(11.9, 1.6, 11.9, 1.1)
    ax.text(0.2, 5.6, "Decision workflow: predictions advise, a human decides", fontsize=11, weight="bold")
    ax.text(0.2, 0.5, "Actors who can view results: Reserver (own prediction), Facility Admin (queue, dashboard).\n"
            "Actor who can record the final decision: Facility Admin only. The model never auto-rejects.",
            fontsize=8.5, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "decision_workflow.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    use_case_diagram(); workflow(); print("diagrams saved")
