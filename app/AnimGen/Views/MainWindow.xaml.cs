using System.Windows;
using AnimGen.Services;
using AnimGen.ViewModels;

namespace AnimGen.Views;

public partial class MainWindow : Window
{
    public MainViewModel Vm { get; }

    public MainWindow(MainViewModel vm)
    {
        Vm = vm;
        DataContext = vm;
        InitializeComponent();
        vm.SkeletonCheckRequested += (_, _) => CheckSkeleton();
    }

    private void CheckSkeleton_Click(object sender, RoutedEventArgs e) => CheckSkeleton();

    // ---- Step 2 pictures zoomed in (owner, 2026-10-31) ----

    private void ZoomRig_Click(object sender, RoutedEventArgs e) => ShowZoomed(Vm.RigImage, Vm.RigImageName);

    private void ZoomHand_Click(object sender, RoutedEventArgs e) => ShowZoomed(Vm.HandImage, "Bàn tay mở");

    private void ShowZoomed(System.Windows.Media.ImageSource? image, string title)
    {
        if (image == null) return;
        new ImageViewerWindow(image, title) { Owner = this }.ShowDialog();
    }

    // ---- The pose editor (owner, 2026-10-31) ----

    private async void NewPose_Click(object sender, RoutedEventArgs e)
    {
        if (Vm.Api == null) return;
        Vm.ShowBusy("Đang chuẩn bị nhân vật mẫu…");
        try
        {
            PoseEditReply start = await Vm.Api.StartPoseAsync(Vm.Angle, null);
            Vm.ShowReady();
            OpenPoseEditor(start, editing: false);
        }
        catch (BackendException ex)
        {
            Vm.ShowError(ex.Message);
        }
    }

    private async void EditPose_Click(object sender, RoutedEventArgs e)
    {
        if (Vm.Api == null || Vm.SelectedAnimation is not { Custom: true } a) return;
        Vm.ShowBusy("Đang mở tư thế…");
        try
        {
            PoseEditReply start = await Vm.Api.OpenPoseAsync(a.Id);
            Vm.ShowReady();
            OpenPoseEditor(start, editing: true);
        }
        catch (BackendException ex)
        {
            Vm.ShowError(ex.Message);
        }
    }

    private void OpenPoseEditor(PoseEditReply start, bool editing)
    {
        if (Vm.Api == null) return;
        var bases = new List<PoseBase> { new(null, "Đứng thẳng") };
        bases.AddRange(Vm.Animations.Where(a => a.Ready).Select(a => new PoseBase(a.Id, a.NameVi)));
        var dlg = new PoseEditorWindow(Vm.Api, start, bases, editing) { Owner = this };
        dlg.ShowDialog();
        if (dlg.SavedId != null) _ = Vm.PoseSavedAsync(dlg.SavedId);
    }

    private async void DeletePose_Click(object sender, RoutedEventArgs e)
    {
        if (Vm.SelectedAnimation is not { Custom: true } a) return;
        if (MessageBox.Show(this, $"Xoá tư thế “{a.NameVi}”? Không lấy lại được.", Title, MessageBoxButton.YesNo,
                            MessageBoxImage.Question) != MessageBoxResult.Yes) return;
        await Vm.DeletePoseAsync(a.Id);
    }

    private async void CheckSkeleton()
    {
        if (Vm.Api == null || Vm.Skeleton?.Image == null) return;
        try
        {
            var dlg = new SkeletonDialog(Vm.Api, Vm.Skeleton) { Owner = this };
            if (dlg.ShowDialog() == true && dlg.Reply != null) await Vm.ApplySkeletonReplyAsync(dlg.Reply);
        }
        catch (InvalidOperationException ex)
        {
            MessageBox.Show(this, ex.Message, Title, MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
