import torch

MSELoss = torch.nn.MSELoss


class ErrorNetLoss(torch.nn.Module):
    def __init__(self, params):
        super(ErrorNetLoss, self).__init__()
        if 'under_certainty_penalty' in params.dict.keys():
            self.under_certainty_penalty = params.under_certainty_penalty
        else:
            self.under_certainty_penalty = 1.

        if 'over_certainty_penalty' in params.dict.keys():
            self.over_certainty_penalty = params.over_certainty_penalty
        else:
            self.over_certainty_penalty = 1.


    def forward(self, error_prediction, truth):
        (energy_prediction, correct_answer) = truth
        # print correct_answer, energy_prediction, error_prediction

        under_certainty = torch.abs(error_prediction - correct_answer) - torch.abs(energy_prediction - correct_answer)
        over_certainty = -1. * under_certainty

        # print 'under_certainty', under_certainty

        clamped_under_certainty = torch.clamp(under_certainty, min=0.)
        clamped_over_certainty = torch.clamp(over_certainty, min=0.)

        under_certainty_loss = self.under_certainty_penalty * torch.mul(clamped_under_certainty, clamped_under_certainty)
        over_certainty_loss = self.over_certainty_penalty * torch.mul(clamped_over_certainty, clamped_over_certainty)

        # print 'under_certainty_loss', under_certainty_loss
        # print 'over_certainty_loss', over_certainty_loss

        loss = torch.mean(under_certainty_loss + over_certainty_loss)
        # print loss
        return loss
